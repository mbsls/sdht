"""Mixed-frequency dynamic factor for the soft block (state-space).

`DFMSoftFactor` extracts a single common factor from the monthly soft
indicators *and* the quarterly Senior Loan Officer survey (SLOOS) using a
state-space dynamic factor model (statsmodels `DynamicFactorMQ`). The
Kalman filter treats quarterly SLOOS as a monthly series observed only
every third month, so no ad-hoc interpolation or forward-fill is needed --
this is the Banbura--Modugno / NY-Fed-nowcast handling of mixed
frequencies.

Interface mirrors models.soft_factor.SoftFactor so the harness
`feature_prep` hook can swap one for the other:
    sf = DFMSoftFactor(monthly_vars, quarterly_var, ...).fit(raw_ff)
    factor_df = sf.transform(raw_ff)      # column 'soft_f1', reindexed

Real-time discipline
--------------------
Only the FILTERED factor is used (one-sided; uses data up to each month),
never the smoothed factor (two-sided; would peek ahead). The model is
refit at each origin on the as-of frame, so loadings carry no future
information. The quarterly series is read from the as-of feature frame
passed in (already vintage-correct via the cube store).

Sign convention: the factor is flipped so it correlates positively with
`sign_reference` (default Michigan sentiment) on the fitted sample, so
"up = better macro" is stable across recursive refits.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ


class DFMSoftFactor:
    def __init__(
        self,
        monthly_vars: list[str],
        quarterly_var: str = "sloos_ci_tightening",
        factor_order: int = 1,
        sign_reference: str = "consumer_sentiment",
    ):
        self.monthly_vars = list(monthly_vars)
        self.quarterly_var = quarterly_var
        self.factor_order = factor_order
        self.sign_reference = sign_reference
        # filled by fit
        self.res_ = None
        self.sign_ = 1.0
        self._factor_period: Optional[pd.Series] = None  # filtered factor, PeriodIndex(M)

    def fit(self, df: pd.DataFrame) -> "DFMSoftFactor":
        m = df[self.monthly_vars].dropna(how="all").copy()
        if len(m) < 24:
            raise ValueError("Too few monthly rows for the DFM soft factor.")
        m.index = pd.PeriodIndex(m.index, freq="M")

        q = None
        if self.quarterly_var in df.columns:
            qser = df[self.quarterly_var].dropna()
            if len(qser) >= 8:
                qser = qser.resample("QS").last().dropna()
                qser.index = pd.PeriodIndex(qser.index, freq="Q")
                q = qser.to_frame(self.quarterly_var)

        mod = DynamicFactorMQ(
            m, endog_quarterly=q, factors=1,
            factor_orders=self.factor_order, idiosyncratic_ar1=True,
        )
        self.res_ = mod.fit(disp=False)
        fac = self.res_.factors.filtered.iloc[:, 0]

        # sign-align to the reference series on the fitted sample
        if self.sign_reference in m.columns:
            ref = m[self.sign_reference].reindex(fac.index).ffill().bfill()
            if np.corrcoef(fac.values, ref.values)[0, 1] < 0:
                self.sign_ = -1.0
        self._factor_period = self.sign_ * fac
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.res_ is None:
            raise RuntimeError("Call .fit() before .transform().")
        # map the PeriodIndex(M) factor back onto the frame's timestamp index
        fac_ts = self._factor_period.copy()
        fac_ts.index = fac_ts.index.to_timestamp(how="start")
        out = pd.DataFrame(index=df.index)
        out["soft_f1"] = fac_ts.reindex(df.index)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        state = "unfit" if self.res_ is None else f"fit, sign={self.sign_:+.0f}"
        return (f"DFMSoftFactor({state}, monthly={self.monthly_vars}, "
                f"quarterly={self.quarterly_var})")
