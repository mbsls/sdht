"""Giacomini--White (2006) test of conditional predictive ability.

Python port of the MATLAB reference in `other_files/giacomini_white.zip`
(`CPAtest.m`, `GW_stat.m`, `NeweyWest.m`; Raffaella Giacomini, 2003).
The port reproduces the reference exactly for the cases it covers
(unconditional and the GW(2006) default conditional instrument
`h_t = [1, d_t]`), and additionally allows an arbitrary
origin-measurable instrument (e.g. the VIX level) for the conditional
test, which is what the soft-vs-hard project needs.

Reference
---------
Giacomini, R. and H. White (2006), "Tests of Conditional Predictive
Ability," *Econometrica* 74(6), 1545--1578.

Test logic (matches CPAtest.m / GW_stat.m)
------------------------------------------
Loss differential d_t = L1_t - L2_t. Build the regressor matrix
reg_t = h_t * d_{t+tau}, where h_t is the instrument vector measurable
at the forecast origin t. Under H0 of equal conditional predictive
ability, E[reg_t] = 0.

  - tau == 1: GWstat = T * R^2 from regressing a vector of ones on reg
              (the reference's formulation; algebraically the efficient
              GMM Wald statistic with the uncentered second-moment
              weighting matrix).
  - tau  > 1: GWstat = T * zbar' inv(Omega) zbar, with Omega the
              uncentered Newey--West HAC estimator (tau-1 Bartlett lags),
              exactly as in NeweyWest.m.

GWstat ~ chi2(q), q = number of instruments. The reference takes
abs(stat) before the chi2 p-value; we keep that convention.

`sign(mean(d))` tells which model is better: positive mean(L1 - L2)
means model 1 has the larger loss, so model 2 is better.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class GWResult:
    stat: float
    pval: float
    dof: int
    mean_loss_diff: float       # mean(L1 - L2) over the full overlap
    better_model: int           # 2 if mean(L1-L2) > 0 (model 2 better), else 1
    horizon: int
    choice: str
    n: int                      # effective sample after the tau shift / NaN drop

    def __repr__(self) -> str:
        return (
            f"GWResult(choice={self.choice!r}, tau={self.horizon}, "
            f"stat={self.stat:.3f}, p={self.pval:.4f}, dof={self.dof}, "
            f"n={self.n}, better=model{self.better_model})"
        )


def _newey_west(reg: np.ndarray, nlags: int) -> np.ndarray:
    """Uncentered Newey--West HAC covariance, matching NeweyWest.m."""
    n = reg.shape[0]
    omega = reg.T @ reg / n
    for ii in range(1, nlags + 1):
        gamma = reg[ii:].T @ reg[:n - ii] / n
        weight = 1.0 - ii / (nlags + 1)
        omega = omega + weight * (gamma + gamma.T)
    return omega


def gw_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    tau: int = 1,
    choice: str = "cond",
    instruments: Optional[np.ndarray] = None,
) -> GWResult:
    """Giacomini--White conditional predictive ability test.

    Parameters
    ----------
    loss1, loss2 : array-like, shape (T,)
        Per-period losses (squared forecast errors) for the two models,
        aligned by evaluation date.
    tau : int
        Forecast horizon. tau>1 uses the NW HAC variance (tau-1 lags).
    choice : {'cond', 'uncond'}
        'uncond' -> instrument is a constant (Diebold--Mariano).
        'cond'   -> GW. Uses `instruments` if given (a constant is
                    prepended), else the reference default h_t=[1, d_t].
    instruments : array-like, shape (T,) or (T,k), optional
        Conditioning variable(s) known at the forecast origin, aligned
        so element t is known when forecasting t+tau. Only for 'cond'.

    Notes
    -----
    Caller is responsible for passing finite, aligned arrays. Any NaNs
    are dropped pairwise (and, for instruments, row-wise) after the
    tau-shift.
    """
    loss1 = np.asarray(loss1, dtype=float)
    loss2 = np.asarray(loss2, dtype=float)
    if loss1.shape != loss2.shape:
        raise ValueError("loss1 and loss2 must have the same shape.")

    d = loss1 - loss2
    T = d.shape[0]
    mean_d_full = float(np.nanmean(d))
    better = 2 if mean_d_full > 0 else 1

    if tau >= T:
        return GWResult(np.nan, np.nan, 0, mean_d_full, better, tau, choice, 0)

    # Instruments h_t (rows 0..T-tau-1) paired with d_{t+tau}.
    if choice == "uncond":
        H = np.ones((T - tau, 1))
    elif choice == "cond":
        if instruments is None:
            H = np.column_stack([np.ones(T - tau), d[:T - tau]])
        else:
            inst = np.asarray(instruments, dtype=float)
            if inst.ndim == 1:
                inst = inst[:, None]
            if inst.shape[0] != T:
                raise ValueError("instruments must align with loss length T.")
            H = np.column_stack([np.ones((T - tau, 1)), inst[:T - tau]])
    else:
        raise ValueError("choice must be 'cond' or 'uncond'.")

    ld = d[tau:]                      # d_{t+tau}, length T-tau
    reg = H * ld[:, None]             # reg_t = h_t * d_{t+tau}

    mask = np.all(np.isfinite(reg), axis=1)
    reg = reg[mask]
    n = reg.shape[0]
    q = reg.shape[1]
    if n <= q + 1:
        return GWResult(np.nan, np.nan, q, mean_d_full, better, tau, choice, n)

    if tau == 1:
        # GWstat = n * R^2 from regressing ones on reg (reference form).
        ones = np.ones(n)
        beta, *_ = np.linalg.lstsq(reg, ones, rcond=None)
        err = ones - reg @ beta
        ss_res = np.mean(err ** 2)
        ss_tot = np.mean((ones - ones.mean()) ** 2)
        # ss_tot is 0 (ones has no variance); reference uses this exact
        # expression, which yields r2 = 1 - ss_res/0. Guard and fall back
        # to the algebraically-equal quadratic form to avoid div-by-zero.
        if ss_tot == 0:
            zbar = reg.mean(axis=0)
            omega = reg.T @ reg / n
            stat = float(n * zbar @ np.linalg.solve(omega, zbar))
        else:
            stat = float(n * (1 - ss_res / ss_tot))
    else:
        zbar = reg.mean(axis=0)
        omega = _newey_west(reg, tau - 1)
        try:
            stat = float(n * zbar @ np.linalg.solve(omega, zbar))
        except np.linalg.LinAlgError:
            stat = float(n * zbar @ np.linalg.pinv(omega) @ zbar)

    stat = abs(stat)
    pval = float(1.0 - stats.chi2.cdf(stat, q))
    return GWResult(stat, pval, q, mean_d_full, better, tau, choice, n)


def dm_test(loss1: np.ndarray, loss2: np.ndarray, tau: int = 1) -> GWResult:
    """Diebold--Mariano as the unconditional special case of GW."""
    return gw_test(loss1, loss2, tau=tau, choice="uncond")
