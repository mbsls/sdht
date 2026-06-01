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

    def _effective_AR(self, B, G, n, p):
        """Effective AR matrices A_1..A_p at frozen weight G.

        With W = [1, y_{t-1}, ..., y_{t-p}], the coefficient block acting on
        W during forecasting is C = B[:k] + G * B[k:2k] (shape (k, n)).
        A_j[i, l] = C[1 + (j-1)*n + l, i].
        """
        k = 1 + n * p
        C = B[:k] + G * B[k:2 * k]            # (k, n)
        return [C[1 + (j - 1) * n: 1 + j * n, :].T for j in range(1, p + 1)]

    @staticmethod
    def _spectral_radius(AR_list, n, p):
        comp = np.zeros((n * p, n * p))
        comp[:n, :] = np.hstack(AR_list)
        if p > 1:
            comp[n:, : n * (p - 1)] = np.eye(n * (p - 1))
        eig = np.linalg.eigvals(comp)
        return float(np.max(np.abs(eig)))

    def _stabilize(self, B, G, n, p, target=0.98):
        """Shrink the high-stress block toward the low-stress block until the
        effective companion at weight G is stationary.

        Returns (B_adjusted, shrink_lambda, spectral_radius_after). The low
        block is left intact (it is the calm-regime dynamics, estimated off
        plenty of data and reliably stationary); only the data-poor high
        block is shrunk by lambda in [0, 1].
        """
        k = 1 + n * p
        rho0 = self._spectral_radius(self._effective_AR(B, G, n, p), n, p)
        if rho0 <= target or G <= 1e-8:
            return B, 1.0, rho0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            B_try = B.copy()
            B_try[k:2 * k] = mid * B[k:2 * k]
            rho = self._spectral_radius(self._effective_AR(B_try, G, n, p), n, p)
            if rho <= target:
                lo = mid
            else:
                hi = mid
        B_adj = B.copy()
        B_adj[k:2 * k] = lo * B[k:2 * k]
        rho_final = self._spectral_radius(self._effective_AR(B_adj, G, n, p), n, p)
        return B_adj, lo, rho_final

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
        k = W.shape[1]
        n = len(endog)

        # Fit every (gamma, c) candidate, recording SSR.
        candidates = []
        for gamma in self.gamma_grid:
            for c in self.c_grid:
                G = self._logistic(gz, gamma, c)[:, None]
                R = np.hstack([W, G * W, dd[:, None]])
                try:
                    Bfull, *_ = np.linalg.lstsq(R, tgt, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                resid = tgt - R @ Bfull
                ssr = float(np.sum((resid / scale) ** 2))
                candidates.append((ssr, gamma, c, Bfull))
        if not candidates:
            raise RuntimeError("STVAR grid search failed.")
        candidates.sort(key=lambda t: t[0])

        z_origin_std = (z_raw[-1] - self.z_mean_) / self.z_std_
        last_obs = Y[-self.lags:]
        last_date = clean.index[-1]
        freq = pd.infer_freq(clean.index) or "MS"
        ffr_idx = endog.index("ffr") if "ffr" in endog else None
        check_h = 12

        # Walk candidates from lowest SSR; accept the first that satisfies BOTH
        # hard requirements: (i) the effective companion at G_origin is
        # stationary (spectral radius < 1) AFTER the high-block shrinkage, and
        # (ii) the resulting forecast keeps FFR >= 0 over the check horizon.
        # The stabilizer only shrinks the high block, so a candidate whose
        # LOW block is itself explosive cannot be rescued -- it is rejected
        # here. If no candidate qualifies, raise so the harness marks the
        # origin ok=False.
        chosen = None
        for ssr, gamma, c, Bfull in candidates:
            B = Bfull[:2 * k]
            G_origin = float(self._logistic(np.array([z_origin_std]), gamma, c)[0])
            B_adj, lam, rho = self._stabilize(B, G_origin, n, self.lags, target=0.98)
            if rho >= 1.0:
                continue                      # low block explosive; unrescuable
            # provisional state for a forecast probe
            self.endog_vars = endog; self.fitted_lags = self.lags
            self.B_ = B_adj; self.G_origin_ = G_origin
            self._last_obs = last_obs; self._last_date = last_date; self._freq = freq
            fpath = self.forecast(check_h)
            if ffr_idx is not None and (fpath["ffr"].values < 0).any():
                continue
            chosen = (ssr, gamma, c, Bfull, B_adj, lam, rho, G_origin)
            break

        if chosen is None:
            raise ValueError(
                "No (gamma, c) combination is both stationary and keeps "
                "FFR >= 0 over the forecast horizon at this origin."
            )

        _, self.gamma_, self.c_, Bfull, self.B_, self.shrink_lambda_, \
            self.spectral_radius_, self.G_origin_ = chosen
        self.delta_ = Bfull[2 * k]
        self.fitted_lags = self.lags
        self._last_obs = last_obs
        self._last_date = last_date
        self._freq = freq
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
