"""Download FRED series at the right vintage and frequency for the analysis.

This module fetches each series listed in a spec config, applies the per-series
aggregation needed to reach a common target frequency, and then delegates
transformations (log, yoy, etc.) and sample-window selection to `auxfun`.

Real-time vs latest data
------------------------
With `real_time: True` in the config, each series is pulled via
`fred.get_series_first_release(code)`, which returns the value that was first
published for each observation date. That is the canonical "real-time" feature
used in the forecasting literature (Croushore-Stark): it captures roughly what
was known at the time of each release. With `real_time: False`, the latest
revised series is pulled instead.

Caveat: first-release values are a one-shot snapshot. They do NOT track
subsequent revisions of the same observation. For transformations that look
back (e.g. yoy-m), this means the denominator is the first-release value from
12 months prior, not the value as it stood when the numerator was published.
That is a known approximation; the proper fix is to maintain a full vintage
cube and recompute features per forecast origin.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Optional
import warnings

import pandas as pd
from fredapi import Fred

import auxfun
import vintage_store
from load_config import load_config

warnings.simplefilter(action="ignore", category=FutureWarning)


# Frequency-short codes returned by FRED metadata, mapped to pandas offsets.
FRED_FREQ_TO_PANDAS = {
    "D":  "D",
    "W":  "W",
    "BW": "2W",
    "M":  "MS",
    "Q":  "QS",
    "SA": "6MS",
    "A":  "YS",
}


def _detect_native_freq(fred: Fred, code: str, series: pd.Series) -> str:
    """Best-effort native pandas frequency for a FRED series.

    Tries series metadata first, then falls back to pandas frequency inference.
    """
    try:
        info = fred.get_series_info(code)
        short = info.get("frequency_short") or info.get("frequency")
        if short:
            short = str(short).strip()
            if short in FRED_FREQ_TO_PANDAS:
                return FRED_FREQ_TO_PANDAS[short]
    except Exception as e:  # network / unknown series  -> fall back
        print(f"  [warn] could not get metadata for {code}: {e!s}")

    inferred = pd.infer_freq(series.index[: min(len(series), 30)])
    if inferred:
        return inferred
    return "MS"  # final fallback


def _fetch_series(fred: Fred, code: str, real_time: bool) -> pd.Series:
    """Pull either the first-release vintage series or the latest values.

    Falls back to `get_series` if FRED's vintage-date cap (2000) is exceeded,
    which happens for daily series with many years of vintages (e.g. VIX).
    For those series this is acceptable: they are market/policy data with no
    meaningful revisions, so first-release ≈ latest.
    """
    if real_time:
        try:
            s = fred.get_series_first_release(code)
        except ValueError as e:
            if "vintage dates" in str(e):
                print(f"  [info] {code}: too many vintages for first-release API; falling back to latest")
                s = fred.get_series(code)
            else:
                raise
    else:
        s = fred.get_series(code)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _maybe_aggregate(
    s: pd.Series,
    native_freq: str,
    target_freq: str,
    how: str,
) -> pd.Series:
    """If native_freq is strictly finer than target_freq, aggregate; else pass through."""
    if native_freq == target_freq:
        return s
    # Use pandas offset hierarchy via period_range length over a fixed window
    # to detect "finer than": more native periods than target periods in 1 year.
    one_year = pd.date_range("2020-01-01", "2020-12-31", freq=native_freq)
    target_year = pd.date_range("2020-01-01", "2020-12-31", freq=target_freq)
    if len(one_year) > len(target_year):
        return auxfun.aggregate_to_freq(s, target_freq, how)
    return s  # native is coarser-or-equal; let asfreq later reindex


def main(
    config_path: str = "config.yaml",
    secrets_path: Optional[str] = None,
) -> pd.DataFrame:
    cfg = load_config(config_path, secrets_path)
    fred = Fred(api_key=cfg["api_key"])

    real_time = bool(cfg.get("real_time", False))
    target_freq = cfg.get("target_frequency", "MS")
    name_convention = cfg.get("name_convention", "tag")
    cube_dir = Path(cfg.get("cube_dir", "data/vintages/cubes"))
    rebuild_cubes = bool(cfg.get("rebuild_cubes", False))
    snapshot_date = (
        pd.Timestamp(cfg["snapshot_date"]) if cfg.get("snapshot_date")
        else pd.Timestamp.now().normalize()
    )
    all_variables: dict[str, dict[str, Any]] = cfg["data_series"]

    frames: list[pd.Series] = []
    for variable, spec in all_variables.items():
        code = spec["code"]
        print(f"fetching {variable} ({code})...")

        if spec.get("vintage_tracked", False):
            cube_path = cube_dir / f"{code}.parquet"
            if cube_path.exists() and not rebuild_cubes:
                cube = vintage_store.load_cube(cube_path)
            else:
                print(f"  building vintage cube for {code} ...")
                cube = vintage_store.update_cube(fred, code, cube_path)
            raw = vintage_store.snapshot_as_of(cube, snapshot_date)
        else:
            series_real_time = bool(spec.get("real_time", real_time))
            raw = _fetch_series(fred, code, real_time=series_real_time)

        native_freq = _detect_native_freq(fred, code, raw)
        agg_how = spec.get("aggregation", "mean")

        # IMPORTANT: transform on native frequency, then aggregate to target.
        # yoy-m/yoy-q/d12 lag by rows, so they must see the native index.
        transformed = auxfun.transform_series(raw, spec)
        s = _maybe_aggregate(transformed, native_freq, target_freq, agg_how)

        s.name = spec["label"] if name_convention == "label" else variable
        frames.append(s)

    df = pd.concat(frames, axis=1)

    sample_start = pd.to_datetime(cfg["sample_start"]) if cfg.get("sample_start") else None
    sample_end = pd.to_datetime(cfg["sample_end"]) if cfg.get("sample_end") else None

    df = auxfun.apply_sample_window(
        df, sample_start=sample_start, sample_end=sample_end, freq=target_freq
    )

    if cfg.get("store_vintage"):
        outdir = Path(cfg.get("output_dir", "."))
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.index.name = "date"
        out_path = outdir / f"{stamp}.csv"
        df.to_csv(out_path)
        print(f"wrote {out_path}")

    return df


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    df = main(config_file)
    print(f"data downloaded successfully: shape={df.shape}, range={df.index.min().date()}..{df.index.max().date()}")
