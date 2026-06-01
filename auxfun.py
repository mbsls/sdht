"""Data transformation and aggregation helpers."""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---- aggregation -------------------------------------------------------------

AGG_HOWS = {"mean", "last", "first", "sum"}


def aggregate_to_freq(s: pd.Series, freq: str, how: str = "mean") -> pd.Series:
    """Resample a series to `freq` using `how` (mean/last/first/sum).

    Use this when a series' native frequency is finer than the analysis
    frequency (e.g. daily VIX -> monthly).
    """
    if how not in AGG_HOWS:
        raise ValueError(f"Unknown aggregation '{how}'. Expected one of {AGG_HOWS}.")
    grouper = s.resample(freq)
    return getattr(grouper, how)()


# ---- transformations ---------------------------------------------------------

KNOWN_TRANSFORMS = {None, "none", "linear", "log", "yoy-m", "yoy-q", "d12"}


def _apply_accumulate(col: pd.Series, mode: str) -> pd.Series:
    if mode != "period-over-period":
        raise ValueError(f"Unknown accumulate mode '{mode}'.")
    vals = col.to_numpy(copy=True)
    out = np.empty_like(vals, dtype=float)
    out[:] = np.nan
    base = 100.0
    out[0] = base
    for j in range(1, len(vals)):
        if np.isfinite(vals[j]):
            base = base * (1 + vals[j] / 100.0)
        out[j] = base
    return pd.Series(out, index=col.index, name=col.name)


def _apply_transform(col: pd.Series, kind: Optional[str]) -> pd.Series:
    if kind in (None, "none", "linear"):
        return col
    if kind == "log":
        return 100 * np.log(col)
    if kind == "yoy-m":
        return 100 * (col - col.shift(12)) / col.shift(12)
    if kind == "yoy-q":
        return 100 * (col - col.shift(4)) / col.shift(4)
    if kind == "d12":
        return col - col.shift(12)
    raise ValueError(f"Unknown transformation '{kind}'. Expected one of {KNOWN_TRANSFORMS}.")


def transform_series(s: pd.Series, spec: dict[str, Any]) -> pd.Series:
    """Apply accumulate + transformation + interpolation to a single series.

    Must be called BEFORE any cross-series alignment / frequency change, because
    `yoy-m`, `yoy-q`, `d12` shift by *rows*: the lag is meaningful only when the
    series sits on its native frequency.
    """
    out = pd.to_numeric(s, errors="coerce").copy()

    if "accumulate" in spec:
        out = _apply_accumulate(out, spec["accumulate"])

    if "transformation" in spec:
        out = _apply_transform(out, spec["transformation"])

    if "interpolate" in spec:
        out = out.interpolate(spec["interpolate"])

    return out.astype(float)


def apply_sample_window(
    df: pd.DataFrame,
    sample_start: Optional[dt.datetime] = None,
    sample_end: Optional[dt.datetime] = None,
    freq: Optional[str] = "MS",
) -> pd.DataFrame:
    """Restrict to [sample_start, sample_end] and (optionally) enforce target freq."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if freq is not None:
        out = out.asfreq(freq)

    if sample_start is not None:
        if not isinstance(sample_start, (dt.datetime, pd.Timestamp)):
            raise TypeError("sample_start must be a datetime/Timestamp.")
        out = out.loc[out.index >= pd.Timestamp(sample_start)]

    if sample_end is not None:
        if not isinstance(sample_end, (dt.datetime, pd.Timestamp)):
            raise TypeError("sample_end must be a datetime/Timestamp.")
        out = out.loc[out.index <= pd.Timestamp(sample_end)]

    return out


def transform_data(
    df: pd.DataFrame,
    variable_dict: dict[str, dict[str, Any]],
    sample_start: Optional[dt.datetime] = None,
    sample_end: Optional[dt.datetime] = None,
    freq: Optional[str] = "MS",
) -> pd.DataFrame:
    """Apply per-series transforms then enforce a target freq + sample window.

    Equivalent to `transform_series` per column followed by `apply_sample_window`.
    Note: this assumes all columns share the same native frequency. For
    mixed-frequency frames, transform each series on its native index BEFORE
    calling this (or call `transform_series` directly and concat).
    """
    out = pd.DataFrame(
        {c: transform_series(df[c], variable_dict.get(c, {})) for c in df.columns},
        index=df.index,
    )
    return apply_sample_window(out, sample_start=sample_start, sample_end=sample_end, freq=freq)


# ---- inverse (kept for symmetry; only `log` implemented today) ---------------

def inverse_transform(
    df: pd.DataFrame,
    variable_dict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        spec = variable_dict.get(c, {})
        if spec.get("inverse_transform") and spec.get("transformation") == "log":
            out[c] = np.exp(out[c] / 100)
    return out
