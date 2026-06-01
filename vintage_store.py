"""Build and query vintage cubes for FRED series.

A *vintage cube* for a series stores `cube[obs, vint]` = the value of the series
at observation date `obs` as last known on or before the vintage release date
`vint`. Cells before the first release of an observation are NaN. After
forward-filling along the vintage axis, `cube.loc[:, D]` is the as-of-D
snapshot of the entire historical series — which is what a forecaster running
at date D would have seen.

Why a cube
----------
`fred.get_series_first_release` returns one value per observation: the value
as first published. That loses later revisions and is contaminated by
base-year rebasings (the deflator changes between the 2023-Q1 first release
and the 2024-Q1 first release, so YoY computed from first-release levels is a
spurious mix of growth + deflator drift).

A vintage cube lets us re-derive features per forecast origin using values
that are *internally consistent within a single vintage snapshot*: every
observation in the snapshot reflects the methodology in force on that
release date.

Storage
-------
Cubes are saved as parquet. After forward-fill along the vintage axis, the
matrix is highly compressible (long runs of repeated values).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
from fredapi import Fred


# ---- build / save / load -----------------------------------------------------

def build_vintage_cube(fred: Fred, code: str) -> pd.DataFrame:
    """Download all (observation, vintage) pairs for `code` and pivot to a cube.

    Returns a DataFrame indexed by observation date, with vintage release dates
    as columns. Values are forward-filled along the vintage axis so that
    `cube.loc[:, D]` for any vintage date `D` (or any date in-between) gives the
    full historical series as known on `D`.
    """
    df = fred.get_series_all_releases(code)
    df = df.copy()
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["date"] = pd.to_datetime(df["date"])

    cube = df.pivot_table(
        index="date",
        columns="realtime_start",
        values="value",
        aggfunc="first",
    ).sort_index(axis=0).sort_index(axis=1)

    return cube.ffill(axis=1)


def save_cube(cube: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cube.copy()
    out.columns = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in out.columns]
    out.to_parquet(path, compression="snappy")


def load_cube(path: Union[str, Path]) -> pd.DataFrame:
    cube = pd.read_parquet(path)
    cube.columns = pd.to_datetime(cube.columns)
    cube.index = pd.to_datetime(cube.index)
    return cube


def update_cube(
    fred: Fred,
    code: str,
    path: Union[str, Path],
) -> pd.DataFrame:
    """Rebuild and overwrite the cube on disk; cheaper than incremental updates
    given how FRED's release API behaves, and the cube is small."""
    cube = build_vintage_cube(fred, code)
    save_cube(cube, path)
    return cube


# ---- query -------------------------------------------------------------------

def snapshot_as_of(
    cube: pd.DataFrame,
    vintage_date: Union[str, pd.Timestamp],
) -> pd.Series:
    """Return the series snapshot as known on or before `vintage_date`.

    Index is observation date; values are the most recent estimate available
    by `vintage_date`. Observations not yet released by then are NaN.
    """
    vd = pd.Timestamp(vintage_date)
    valid_cols = cube.columns[cube.columns <= vd]
    if len(valid_cols) == 0:
        return pd.Series(index=cube.index, dtype=float, name=str(vd.date()))
    snap = cube[valid_cols[-1]].copy()
    snap.name = str(vd.date())
    return snap


def first_release_series(cube: pd.DataFrame) -> pd.Series:
    """For each observation date, the value as first published.

    Equivalent to `fred.get_series_first_release(code)` but derived from the
    cube without re-hitting the API.
    """
    # The first vintage column in which each row has a non-NaN entry holds
    # the first-release value for that row (because we ffill along columns,
    # the value persists once published).
    out = {}
    for obs in cube.index:
        row = cube.loc[obs]
        non_null = row.dropna()
        if not non_null.empty:
            out[obs] = non_null.iloc[0]
    return pd.Series(out).sort_index()
