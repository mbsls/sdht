"""Pseudo-real-time forecasting harness.

For each origin date, build the as-of feature frame, fit the model on it,
forecast `max(horizons)` steps ahead, and record one row per
(origin, spec, target, horizon) tuple with the forecast value and the
realized value (looked up from the cubes/static as a separate query).

The harness is model-agnostic: any object exposing `.fit(df, variables)`
and `.forecast(steps) -> DataFrame[steps × variables]` plugs in.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Protocol

import pandas as pd
from tqdm.auto import tqdm

from evaluation.feature_frame import build_feature_frame, get_realization


class _Model(Protocol):
    def fit(self, df: pd.DataFrame, variables: list[str]) -> "_Model": ...
    def forecast(self, steps: int) -> pd.DataFrame: ...


def _identity_prep(raw_ff: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    return raw_ff


def run_pseudo_real_time(
    model_factory: Callable[[], _Model],
    cfg: dict,
    cubes: dict[str, pd.DataFrame],
    static: dict[str, pd.Series],
    origins: Iterable[pd.Timestamp],
    variables: list[str],
    targets: list[str],
    horizons: list[int],
    spec_name: str,
    feature_prep: Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame] = _identity_prep,
    min_obs: Optional[int] = None,
    realization_basis: str = "latest",
    progress: bool = True,
) -> pd.DataFrame:
    """Run the loop and return a long-format forecasts DataFrame.

    Parameters
    ----------
    variables : list[str]
        Raw config-side variables to pull into the as-of feature frame.
    feature_prep : callable
        Optional transformation `(raw_ff, origin) -> final_ff` applied before
        fitting. The default is identity. Use this to introduce derived
        features (e.g. a recursively-refit PCA factor) without bloating the
        harness with model-specific paths.
    targets : list[str]
        Names of forecast targets. Must be present in the *final* feature
        frame returned by `feature_prep`.

    Columns of returned DataFrame:
        origin, spec, target, horizon, forecast, realized, error,
        n_obs, fit_lags, ok
    `error` = forecast - realized.
    `ok=False` indicates the origin was skipped (typically not enough data).
    """
    rows: list[dict] = []
    max_h = max(horizons)
    origins = list(pd.DatetimeIndex(origins))
    min_obs_eff = 36 if min_obs is None else min_obs

    iterable = tqdm(origins, desc=f"{spec_name}") if progress else origins

    def fail_rows(origin, n_obs):
        return [
            {
                "origin": origin, "spec": spec_name, "target": target,
                "horizon": h, "forecast": float("nan"),
                "realized": float("nan"), "error": float("nan"),
                "n_obs": n_obs, "fit_lags": None, "ok": False,
            }
            for target in targets for h in horizons
        ]

    for origin in iterable:
        raw_ff = build_feature_frame(cfg, cubes, static, origin, variables=variables)
        try:
            final_ff = feature_prep(raw_ff, origin)
        except Exception:
            rows.extend(fail_rows(origin, len(raw_ff)))
            continue

        clean = final_ff.dropna(how="any")
        model_vars = list(clean.columns)

        if len(clean) < min_obs_eff or not all(t in model_vars for t in targets):
            rows.extend(fail_rows(origin, len(clean)))
            continue

        model = model_factory()
        try:
            model.fit(clean, model_vars)
            f = model.forecast(max_h)
        except Exception:
            rows.extend(fail_rows(origin, len(clean)))
            continue

        for target in targets:
            for h in horizons:
                fcast_date = f.index[h - 1]
                yhat = float(f.loc[fcast_date, target])
                yreal = get_realization(
                    cfg, cubes, static, target, fcast_date, basis=realization_basis,
                )
                rows.append({
                    "origin": origin, "spec": spec_name, "target": target,
                    "horizon": h, "forecast": yhat, "realized": yreal,
                    "error": yhat - yreal if pd.notna(yreal) else float("nan"),
                    "n_obs": len(clean),
                    "fit_lags": getattr(model, "fitted_lags", None),
                    "ok": True,
                })

    return pd.DataFrame(rows)


def monthly_origins(start: str, end: str) -> pd.DatetimeIndex:
    """Convenience: month-start dates between start and end inclusive."""
    return pd.date_range(start=start, end=end, freq="MS")
