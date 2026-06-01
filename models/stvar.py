"""Logistic smooth-transition VAR (STVAR) with VIX as an EXOGENOUS
transition variable, plus a COVID intercept dummy.

Model
-----
    y_t = [W_t, G(z_{t-1}) * W_t] @ B + delta * D_t + eps_t,
    W_t = [1, y_{t-1}, ..., y_{t-p}],
    G(z) = 1 / (1 + exp(-gamma * (z - c))),

where:
  * y is the endogenous block (hard, or hard+soft variables);
  * z is the EXOGENOUS transition variable (standardized VIX), lagged one
    period so it is known at the forecast origin;
  * D_t is the COVID intercept dummy (1 over a known window, else 0).

Why VIX is exogenous (lesson from the first attempt)
---------------------------------------------------
Putting VIX in the endogenous block makes the system forecast VIX; its
forecasts then drive the regime weight G, which feeds back into ever
wilder forecasts and the iteration diverges. Treating VIX purely as an
observed conditioning variable removes the feedback loop.

Frozen-regime forecasting
-------------------------
Future VIX is unknown, so we hold the regime weight fixed at its value
at the forecast origin, G(z_T), for the whole horizon (the standard
Auerbach--Gorodnichenko convention). A forecast made in a high-VIX month
therefore uses high-stress dynamics throughout its path, which is exactly
the regime-conditional behavior we want to test. The COVID dummy is set
to zero over the forecast horizon, so it only cleans up in-sample
estimation (as in models/covid_dummy_var.py).

Estimation
----------
Conditional on (gamma, c) the model is linear; B is OLS equation by
equation on [W, G*W, D]. We grid-search (gamma, c) to minimize total SSR
on standardized residuals.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class STVAR:
    def __init__(
        self,
        lags: int = 4,
        transition_var: str = "vix",
        covid_window: tuple[str, str] = ("2020-03-01", "2021-06-01"),
        gamma_grid: Optional[list[float]] = None,
        c_grid: Optional[list[float]] = None,
    ):
        self.lags = lags
        self.transition_var = transition_var
        self.covid_window = (pd.Timestamp(covid_window[0]), pd.Timestamp(covid_window[1]))
        self.gamma_grid = gamma_grid if gamma_grid is not None else [1.0, 2.0, 4.0, 7.0, 12.0]
        self.c_grid = c_grid if c_grid is not None else [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        # filled by fit
        self.endog_vars: Optional[list[str]] = None
        self.fitted_lags: Optional[int] = None
        self.B_: Optional[np.ndarray] = None
        self.delta_: Optional[np.ndarray] = None
        self.gamma_: Optional[float] = None
        self.c_: Optional[float] = None
        self.G_origin_: Optional[float] = None
        self.z_mean_: Optional[float] = None
        self.z_std_: Optional[float] = None
        self._last_obs: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._freq: Optional[str] = None

    @staticmethod
    def _logistic(z_std, gamma, c):
        return 1.0 / (1.0 + np.exp(-gamma * (z_std - c)))

    def _build(self, Y, z_std, dummy):
        """Stack regressors for t = p..T-1.

        Returns W (incl const), target y_t, g = G-input z_{t-1}, d = D_t.
        """
        T, n = Y.shape
        p = self.lags
        W, tgt, gz, dd = [], [], [], []
        for t in range(p, T):
            w = [1.0]
            for k in range(1, p + 1):
                w.extend(Y[t - k])
            W.append(w); tgt.append(Y[t]); gz.append(z_std[t - 1]); dd.append(dummy[t])
        return np.asarray(W), np.asarray(tgt), np.asarray(gz), np.asarray(dd)

    def fit(self, df: pd.DataFrame, variables: list[str]) -> "STVAR":
        if self.transition_var not in df.columns:
            raise ValueError(f"transition_var '{self.transition_var}' not in df.")
        # endog = requested variables MINUS the transition var (it's exogenous)
        endog = [v for v in variables if v != self.transition_var]
        cols = endog + [self.transition_var]
        clean = df[cols].dropna(how="any")
        if len(clean) < self.lags + 12:
            raise ValueError("Too few complete-case rows for STVAR.")
        self.endog_vars = endog

        Y = clean[endog].values
        z_raw = clean[self.transition_var].values
        self.z_mean_ = float(z_raw.mean()); self.z_std_ = float(z_raw.std()) or 1.0
        z_std = (z_raw - self.z_mean_) / self.z_std_

        dummy = ((clean.index >= self.covid_window[0]) &
                 (clean.index <= self.covid_window[1])).astype(float)

        W, tgt, gz, dd = self._build(Y, z_std, dummy)
        scale = tgt.std(axis=0); scale[scale == 0] = 1.0

        best = None
        for gamma in self.gamma_grid:
            for c in self.c_grid:
                G = self._logistic(gz, gamma, c)[:, None]
                R = np.hstack([W, G * W, dd[:, None]])
                try:
                    B, *_ = np.linalg.lstsq(R, tgt, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                resid = tgt - R @ B
                ssr = float(np.sum((resid / scale) ** 2))
                if best is None or ssr < best[0]:
                    best = (ssr, gamma, c, B)
        if best is None:
            raise RuntimeError("STVAR grid search failed.")
        _, self.gamma_, self.c_, Bfull = best
        k = W.shape[1]
        self.B_ = Bfull[:2 * k]          # AR + interaction blocks
        self.delta_ = Bfull[2 * k]       # COVID dummy row
        self.fitted_lags = self.lags

        # frozen regime weight at the origin (last observed z)
        z_origin_std = (z_raw[-1] - self.z_mean_) / self.z_std_
        self.G_origin_ = float(self._logistic(np.array([z_origin_std]), self.gamma_, self.c_)[0])

        self._last_obs = Y[-self.lags:]
        self._last_date = clean.index[-1]
        self._freq = pd.infer_freq(clean.index) or "MS"
        return self

    def forecast(self, steps: int) -> pd.DataFrame:
        if self.B_ is None:
            raise RuntimeError("Call .fit() before .forecast().")
        n = len(self.endog_vars); p = self.lags
        G = self.G_origin_                       # frozen regime; future D = 0
        history = [row.copy() for row in self._last_obs]
        out = np.empty((steps, n))
        for h in range(steps):
            w = [1.0]
            for k in range(1, p + 1):
                w.extend(history[-k])
            w = np.asarray(w)
            r = np.concatenate([w, G * w])
            y_next = r @ self.B_
            history.append(y_next); out[h] = y_next
        future = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=steps, freq=self._freq,
        )
        return pd.DataFrame(out, index=future, columns=self.endog_vars)

    # keep a `variables`/`fitted_lags` surface compatible with the harness
    @property
    def variables(self):
        return self.endog_vars

    def transition_weights(self, df: pd.DataFrame) -> pd.Series:
        clean = df[[self.transition_var]].dropna()
        z_std = (clean[self.transition_var].values - self.z_mean_) / self.z_std_
        return pd.Series(self._logistic(z_std, self.gamma_, self.c_),
                         index=clean.index, name="G_high_stress")

    def __repr__(self) -> str:
        if self.B_ is None:
            return f"STVAR(unfit, lags={self.lags}, transition={self.transition_var})"
        return (f"STVAR(p={self.fitted_lags}, n_endog={len(self.endog_vars)}, "
                f"gamma={self.gamma_}, c={self.c_}, G_origin={self.G_origin_:.3f})")
