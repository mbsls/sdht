"""Univariate Markov-switching AR with a hand-rolled real-time forecast.

statsmodels' MarkovAutoregression fits the model and gives filtered regime
probabilities and the transition matrix, but its out-of-sample `predict` is
NotImplementedError. This wrapper adds an iterated multi-step forecast that
respects the real-time information set:

  * fit on data through the origin;
  * take the FILTERED (not smoothed) regime probability at the origin
    -- smoothed probabilities use future data and would leak;
  * propagate the regime distribution forward with the transition matrix
    and form the regime-weighted point forecast at each step.

Design choices (motivated by the M4 STVAR failures):
  * AR coefficients are COMMON across regimes (switching_ar=False); only
    the intercept and error variance switch. With ~80-350 monthly obs and
    a univariate target this is what is actually estimable, and it keeps
    the dynamics stationary as long as the single AR polynomial is.
  * Optional exogenous regressors (e.g. a soft factor) enter with common
    coefficients, so "does soft data help" is a clean nested comparison.

The regime is LATENT (inferred from the data), so unlike the STVAR it needs
no VIX threshold -- a genuinely different identification of "hard times".
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


class MSAR:
    def __init__(self, order: int = 2, k_regimes: int = 2,
                 switching_variance: bool = True):
        self.order = order
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        # filled by fit
        self.res_ = None
        self.target_: Optional[str] = None
        self.exog_cols_: list[str] = []
        self.const_: Optional[np.ndarray] = None      # (k,) regime intercepts
        self.ar_: Optional[np.ndarray] = None         # (order,) shared AR
        self.exog_beta_: Optional[np.ndarray] = None  # (n_exog,) shared
        self.P_: Optional[np.ndarray] = None          # (k,k), cols sum to 1
        self.p_origin_: Optional[np.ndarray] = None   # (k,) filtered prob at origin
        self._last_y: Optional[np.ndarray] = None     # last `order` y values
        self._last_exog: Optional[np.ndarray] = None  # exog row at origin (held flat)
        self._last_date = None
        self._freq = None
        self.fitted_lags = None

    def fit(self, df: pd.DataFrame, variables: list[str]) -> "MSAR":
        """variables[0] is the target; the rest are exogenous regressors."""
        self.target_ = variables[0]
        self.exog_cols_ = list(variables[1:])
        clean = df[variables].dropna(how="any")
        y = clean[self.target_].values
        exog = clean[self.exog_cols_].values if self.exog_cols_ else None

        mod = MarkovAutoregression(
            y, k_regimes=self.k_regimes, order=self.order,
            exog=exog, switching_ar=False,
            switching_variance=self.switching_variance,
        )
        # EM multi-start: statsmodels' fit() random-searches starting values
        # (search_reps) before the quasi-Newton step. Adding exogenous
        # regressors (e.g. UMCSENT) otherwise gives frequent poor fits from a
        # bad default start. Keep the best-likelihood fit with FINITE llf and
        # params -- the strict `converged` flag is unreliable here (statsmodels
        # often reports converged=False on usable fits), so we screen on
        # finiteness, not the flag. Raise only if every start gives nan
        # (genuinely degenerate, e.g. core PCE + a regressor).
        best = None
        for reps, em in [(20, 10), (50, 20)]:
            try:
                r = mod.fit(search_reps=reps, em_iter=em, maxiter=200, disp=0)
            except Exception:
                continue
            if np.isfinite(r.llf) and np.all(np.isfinite(r.params)):
                if best is None or r.llf > best.llf:
                    best = r
        if best is None:
            raise RuntimeError("MS-AR failed to produce a finite fit from any start.")
        self.res_ = best
        names = list(self.res_.model.param_names)
        p = self.res_.params

        # In Hamilton's MarkovAutoregression the model is in MEAN-DEVIATION
        # form: const[j] is the regime MEAN mu_j, and the AR operates on
        # (y_{t-l} - mu). Exogenous regressors enter the conditional mean and
        # are named 'x{i}[0]' (statsmodels appends a regime index even when
        # non-switching -> a single set repeated across the index).
        self.const_ = np.array([p[names.index(f"const[{j}]")] for j in range(self.k_regimes)])
        self.ar_ = np.array([p[names.index(f"ar.L{l}")] for l in range(1, self.order + 1)])
        if self.exog_cols_:
            import re
            beta = []
            for i in range(len(self.exog_cols_)):
                # statsmodels names exog params 'x{i+1}' optionally followed by
                # a regime index like '[1]'. Match either form.
                pat = re.compile(rf"^x{i+1}(\[\d+\])?$")
                idx = next((j for j, nm in enumerate(names) if pat.match(nm)), None)
                if idx is None:
                    raise KeyError(f"exog param x{i+1} not found in {names}")
                beta.append(p[idx])
            self.exog_beta_ = np.array(beta)
        else:
            self.exog_beta_ = np.array([])

        P = self.res_.regime_transition
        self.P_ = P[:, :, 0] if P.ndim == 3 else P
        # filtered (not smoothed) regime probabilities -- real-time safe.
        filt = np.asarray(self.res_.filtered_marginal_probabilities)
        self.p_origin_ = filt[-1]
        # regime-weighted mean at the last `order` observations, for the
        # mean-deviation AR recursion: dev_{t-l} = y_{t-l} - mu_bar_{t-l}.
        mu_recent = filt[-self.order:] @ self.const_      # (order,)
        self._dev_hist = list(y[-self.order:] - mu_recent)  # oldest..newest

        self._last_y = y[-self.order:]
        self._last_exog = exog[-1] if exog is not None else None
        self._last_date = clean.index[-1]
        self._freq = pd.infer_freq(clean.index) or "MS"
        self.fitted_lags = self.order
        return self

    def forecast(self, steps: int) -> pd.DataFrame:
        if self.res_ is None:
            raise RuntimeError("Call .fit() before .forecast().")
        # Mean-deviation form: dev_t = sum_l ar_l * dev_{t-l} + eps, and
        # y_t = mu_bar_t + dev_t + exog_term, where mu_bar_t is the
        # regime-probability-weighted mean at step t. The AR acts on the
        # stationary deviation, so it decays; the level path follows the
        # forecast mean. Exog held flat over the horizon (last observed row).
        dev_hist = list(self._dev_hist)        # oldest..newest deviations
        p = self.p_origin_.copy()
        exog_term = (float(self._last_exog @ self.exog_beta_)
                     if self.exog_cols_ else 0.0)
        out = np.empty(steps)

        for h in range(steps):
            p = self.P_ @ p                    # propagate regime distribution
            mu_bar = float(p @ self.const_)    # forecast regime-weighted mean
            dev_next = float(np.dot(self.ar_, dev_hist[-1::-1][:self.order]))
            dev_hist.append(dev_next)
            out[h] = mu_bar + dev_next + exog_term

        idx = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=steps, freq=self._freq,
        )
        return pd.DataFrame({self.target_: out}, index=idx)

    def regime_probabilities(self) -> pd.Series:
        """Smoothed prob of the high-variance regime over the fitted sample
        (for plotting only -- uses full-sample info, not for forecasting)."""
        sm = np.asarray(self.res_.smoothed_marginal_probabilities)
        hi = int(np.argmax(self.res_.params[[self.res_.model.param_names.index(f'sigma2[{j}]')
                                             for j in range(self.k_regimes)]]))
        return pd.Series(sm[:, hi], name=f"P(high-var regime={hi})")

    def __repr__(self) -> str:
        if self.res_ is None:
            return f"MSAR(unfit, order={self.order}, k={self.k_regimes})"
        return (f"MSAR(order={self.order}, k={self.k_regimes}, "
                f"target={self.target_}, exog={self.exog_cols_}, "
                f"p_origin={self.p_origin_.round(2).tolist()})")
