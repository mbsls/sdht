"""Thin wrapper around statsmodels VAR.

The harness expects a model object with `fit(df, variables)` and
`forecast(steps)`. This module provides that for an ordinary VAR with
fixed lag order or AIC/BIC-selected lag order.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


class LinearVAR:
    """Fixed-lag VAR with a fit/forecast interface.

    Parameters
    ----------
    lags : int or 'aic' or 'bic'
        Lag order. Pass 'aic' or 'bic' to select via information criterion
        with `max_lags` as the upper bound.
    max_lags : int
        Upper bound for IC-based lag selection. Ignored if `lags` is an int.
    trend : str
        Passed through to statsmodels VAR.fit; 'c' (constant), 'ct'
        (constant + trend), 'n' (no determinstic terms).
    """

    def __init__(
        self,
        lags: Union[int, str] = 4,
        max_lags: int = 12,
        trend: str = "c",
    ):
        self.lags_spec = lags
        self.max_lags = max_lags
        self.trend = trend
        self.fitted_lags: Optional[int] = None
        self.variables: Optional[list[str]] = None
        self.result = None
        self._last_obs: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._freq: Optional[str] = None

    def fit(self, df: pd.DataFrame, variables: list[str]) -> "LinearVAR":
        clean = df[variables].dropna(how="any")
        if clean.empty:
            raise ValueError("No complete-case rows for the requested variables.")
        self.variables = list(clean.columns)
        model = VAR(clean)
        if isinstance(self.lags_spec, int):
            self.fitted_lags = self.lags_spec
        else:
            sel = model.select_order(maxlags=self.max_lags)
            criterion = self.lags_spec.lower()
            self.fitted_lags = int(getattr(sel, criterion))
            self.fitted_lags = max(self.fitted_lags, 1)
        self.result = model.fit(self.fitted_lags, trend=self.trend)
        self._last_obs = clean.values[-self.fitted_lags:]
        self._last_date = clean.index[-1]
        self._freq = clean.index.freqstr or pd.infer_freq(clean.index)
        return self

    def forecast(self, steps: int) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call .fit() before .forecast().")
        f = self.result.forecast(self._last_obs, steps=steps)
        # Build the future index at the inferred frequency.
        future = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(self._freq or "MS"),
            periods=steps,
            freq=self._freq or "MS",
        )
        return pd.DataFrame(f, index=future, columns=self.variables)

    def __repr__(self) -> str:
        if self.fitted_lags is None:
            return f"LinearVAR(unfit, lags={self.lags_spec})"
        return (
            f"LinearVAR(p={self.fitted_lags}, vars={self.variables}, "
            f"T={len(self._last_obs)+self.fitted_lags})"
        )
