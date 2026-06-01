"""Build feature DataFrames respecting real-time information sets.

Given the dual-window data spec (configs/us.yaml), a set of vintage cubes,
and a set of static (non-revising) series, `build_feature_frame(as_of)`
returns the DataFrame of features that a forecaster operating on `as_of`
could legitimately have used.

The function does the obvious right thing per series:
  * vintage_tracked series → slice the cube via snapshot_as_of(as_of)
  * non-tracked series     → restrict to dates strictly before as_of
Then applies the per-series transformation on the native frequency,
optionally aggregates to the target frequency, and concatenates.

The `as_of` convention is the START of the forecast origin period
(e.g. as_of='2020-05-01' means "forecaster operating at the start of
May 2020, having data through April 2020"). Daily/weekly aggregations
done before the as_of filter therefore include the full prior month.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

import auxfun
import vintage_store


def load_inputs(
    cfg: dict,
    fred,
    cube_dir: Union[str, Path] = "data/vintages/cubes",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """One-shot load of the cubes (vintage-tracked) and static series.

    Returns (cubes, static), both keyed by FRED code.
    """
    cube_dir = Path(cube_dir)
    cubes: dict[str, pd.DataFrame] = {}
    static: dict[str, pd.Series] = {}
    global_real_time = bool(cfg.get("real_time", False))

    for var, spec in cfg["data_series"].items():
        code = spec["code"]
        if spec.get("vintage_tracked"):
            cubes[code] = vintage_store.load_cube(cube_dir / f"{code}.parquet")
        else:
            real_time = bool(spec.get("real_time", global_real_time))
            if real_time:
                try:
                    s = fred.get_series_first_release(code)
                except ValueError as e:
                    if "vintage dates" not in str(e):
                        raise
                    s = fred.get_series(code)
            else:
                s = fred.get_series(code)
            s.index = pd.to_datetime(s.index)
            static[code] = s.sort_index()
    return cubes, static


def build_feature_frame(
    cfg: dict,
    cubes: dict[str, pd.DataFrame],
    static: dict[str, pd.Series],
    as_of: Union[str, pd.Timestamp],
    variables: Optional[list[str]] = None,
) -> pd.DataFrame:
    """As-of feature frame at the target frequency.

    `variables` lets you restrict to a subset (e.g. hard-only spec).
    """
    target_freq = cfg.get("target_frequency", "MS")
    series_specs = cfg["data_series"]
    if variables is None:
        variables = list(series_specs.keys())

    as_of = pd.Timestamp(as_of)

    frames: list[pd.Series] = []
    for var in variables:
        spec = series_specs[var]
        code = spec["code"]

        if spec.get("vintage_tracked"):
            raw = vintage_store.snapshot_as_of(cubes[code], as_of)
        else:
            raw = static[code].copy()

        # Native-frequency transformation first.
        transformed = auxfun.transform_series(raw, spec)

        # If the series is finer than target freq, aggregate.
        if "aggregation" in spec:
            transformed = auxfun.aggregate_to_freq(
                transformed, target_freq, spec["aggregation"]
            )

        # Real-time cut: only observations dated strictly before the origin.
        transformed = transformed.loc[transformed.index < as_of]
        transformed.name = var
        frames.append(transformed)

    df = pd.concat(frames, axis=1)
    sample_start = (
        pd.to_datetime(cfg["sample_start"]) if cfg.get("sample_start") else None
    )
    df = auxfun.apply_sample_window(df, sample_start=sample_start, freq=target_freq)
    return df


def get_realization(
    cfg: dict,
    cubes: dict[str, pd.DataFrame],
    static: dict[str, pd.Series],
    target: str,
    obs_date: Union[str, pd.Timestamp],
    basis: str = "latest",
) -> float:
    """Realized value of `target` at `obs_date` for forecast-error scoring.

    basis: 'latest' (most recent revised value) or 'first_release'
    (the value as first published). 'latest' is the default convention
    for forecast evaluation.
    """
    spec = cfg["data_series"][target]
    code = spec["code"]
    obs_date = pd.Timestamp(obs_date)
    target_freq = cfg.get("target_frequency", "MS")

    if spec.get("vintage_tracked"):
        cube = cubes[code]
        if basis == "latest":
            raw = cube.iloc[:, -1]
        elif basis == "first_release":
            raw = vintage_store.first_release_series(cube)
        else:
            raise ValueError(f"unknown basis '{basis}'")
    else:
        raw = static[code]

    s = auxfun.transform_series(raw, spec)
    if "aggregation" in spec:
        s = auxfun.aggregate_to_freq(s, target_freq, spec["aggregation"])

    if obs_date not in s.index:
        return float("nan")
    val = s.loc[obs_date]
    return float(val) if pd.notna(val) else float("nan")
