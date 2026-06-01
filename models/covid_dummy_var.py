"""VAR with a known-event dummy + first-lag interaction.

Implements the model

    y_t = c + sum_{k=1}^{p} A_k y_{t-k}
            + delta * D_t
            + B (D_t * y_{t-1})
            + eps_t

where D_t is an exogenous 0/1 indicator for a known regime window
(e.g. the COVID period). The dummy enters additively as an intercept
shift; its product with the first lag of the endogenous block enters as
a vector of interaction terms, so that COVID-period AR-1 dynamics can
differ from tranquil-time dynamics without committing to a full
regime-switching model.

Forecasting
-----------
At any forecast origin, future D is set to zero (we assume the regime
window has ended). Hence the iterated forecast uses only the
tranquil-time matrices A_k and intercept c. The interaction
coefficients only contribute *in-sample* to keep the tranquil-time
estimates uncontaminated by the regime episode.

This is a fixed-regime cousin of the data-driven STVAR / TVAR
specifications that come later in the project.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


class CovidDummyVAR:
    def __init__(
        self,
        lags: int = 4,
        trend: str = "c",
        window: tuple[str, str] = ("2020-03-01", "2021-06-01"),
    ):
        self.lags = lags
        self.trend = trend
        self.window = (pd.Timestamp(window[0]), pd.Timestamp(window[1]))
        # populated by fit
        self.variables: Optional[list[str]] = None
        self.fitted_lags: Optional[int] = None
        self.result_ = None
        self.intercept_: Optional[np.ndarray] = None        # (n,)
        self.coefs_:     Optional[np.ndarray] = None        # (p, n, n)
        self.exog_params_: Optional[np.ndarray] = None      # (n_exog, n)
        self._last_obs: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._freq: Optional[str] = None

    # ---- exog construction -------------------------------------------------

    def _build_exog(self, endog: pd.DataFrame) -> pd.DataFrame:
        dummy = pd.Series(0.0, index=endog.index, name="covid_d")
        in_window = (endog.index >= self.window[0]) & (endog.index <= self.window[1])
        dummy.loc[in_window] = 1.0

        lag1 = endog.shift(1)
        inter = lag1.multiply(dummy, axis=0)
        inter.columns = [f"covid_x_{c}_lag1" for c in endog.columns]
        return pd.concat([dummy.to_frame(), inter], axis=1)

    # ---- fit ---------------------------------------------------------------

    def fit(self, df: pd.DataFrame, variables: list[str]) -> "CovidDummyVAR":
        endog = df[variables]
        exog = self._build_exog(endog)
        joined = pd.concat([endog, exog], axis=1).dropna(how="any")
        if joined.empty:
            raise ValueError("No complete-case rows after building exog.")
        self.variables = list(variables)
        endog_clean = joined[variables]
        exog_clean = joined.drop(columns=variables)
        self._exog_cols = list(exog_clean.columns)

        model = VAR(endog_clean.values, exog=exog_clean.values)
        self.result_ = model.fit(self.lags, trend=self.trend)
        self.fitted_lags = self.lags

        # Pull out the AR coefficients and exog params in a form we can use
        # for our own iterated forecast (statsmodels' VAR.forecast does not
        # accept future exog).
        # result.coefs is (lags, n, n) — AR matrices in row form
        self.coefs_ = np.asarray(self.result_.coefs)
        # intercept and exog params live in result.params (constants + exog).
        # statsmodels' VARResults.params has shape (k_trend + n*p + k_exog, n).
        # The "trend" rows are first; "exog" rows are last.
        n = endog_clean.shape[1]
        params = np.asarray(self.result_.params)
        k_trend = {"n": 0, "c": 1, "ct": 2, "ctt": 3}.get(self.trend, 1)
        # First k_trend rows: trend terms. For 'c', that's a constant.
        if k_trend > 0:
            self.intercept_ = params[0, :].copy() if self.trend == "c" else params[:k_trend, :].sum(axis=0)
        else:
            self.intercept_ = np.zeros(n)
        # Exog params are the last k_exog rows
        k_exog = exog_clean.shape[1]
        self.exog_params_ = params[-k_exog:, :].copy()

        self._last_obs = endog_clean.values[-self.lags:]
        self._last_date = endog_clean.index[-1]
        self._freq = pd.infer_freq(endog_clean.index) or "MS"
        return self

    # ---- forecast ----------------------------------------------------------

    def forecast(self, steps: int) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("Call .fit() before .forecast().")
        n = len(self.variables)
        p = self.fitted_lags

        # Future exog: dummy = 0 (post-COVID) -> all interactions = 0.
        # So the exog contribution vanishes and we iterate the AR only.
        history = list(self._last_obs)  # list of length p, oldest first
        out = np.empty((steps, n))
        for h in range(steps):
            y_next = self.intercept_.copy()
            for k in range(p):
                y_next = y_next + self.coefs_[k] @ history[-(k + 1)]
            history.append(y_next)
            out[h] = y_next

        future = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=steps,
            freq=self._freq,
        )
        return pd.DataFrame(out, index=future, columns=self.variables)

    # ---- debug helpers -----------------------------------------------------

    @property
    def covid_intercept_shift(self) -> Optional[pd.Series]:
        """Coefficient on the dummy itself, per equation."""
        if self.exog_params_ is None:
            return None
        return pd.Series(self.exog_params_[0], index=self.variables, name="covid_d")

    @property
    def covid_lag1_interaction(self) -> Optional[pd.DataFrame]:
        """Coefficients on the (dummy * lag-1 of variable i) interactions.

        Rows: equation (variable being predicted).
        Columns: which lagged variable the interaction multiplies.
        """
        if self.exog_params_ is None:
            return None
        # First exog column is the dummy itself, rest are interactions
        inter = self.exog_params_[1:, :].T  # rows = equations, cols = interactions
        return pd.DataFrame(inter, index=self.variables,
                            columns=[c.replace("covid_x_", "").replace("_lag1", "")
                                     for c in self._exog_cols[1:]])

    def __repr__(self) -> str:
        if self.fitted_lags is None:
            return f"CovidDummyVAR(unfit, lags={self.lags})"
        return (
            f"CovidDummyVAR(p={self.fitted_lags}, vars={self.variables}, "
            f"window={self.window[0].date()}..{self.window[1].date()})"
        )
