"""Re-run the soft-FACTOR specs with the mixed-frequency DFM factor.

Only the hard_plus_soft_factor spec changes (the factor now includes
quarterly SLOOS via a state-space DFM). Hard-only and UMCSENT specs are
unchanged, so we reuse those from the existing PCA runs and only compute
the DFM-factor arm here, writing to *_dfmfactor.parquet. Short window
only (the rich soft block is short-window).

Covers both models: STVAR (VIX-threshold) and MS-AR (latent regime).
Also records the per-origin factor sign to check stability.
"""
import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from load_config import load_config
from evaluation.feature_frame import build_feature_frame, get_realization
from evaluation.harness import run_pseudo_real_time, monthly_origins
from models.stvar import STVAR
from models.msar import MSAR
from models.dfm_soft_factor import DFMSoftFactor

cfg = load_config('configs/us.yaml')
cubes, static = pickle.load(open('data/cache/inputs.pkl', 'rb')).values()

HARD = ['unemployment', 'industrial_production', 'payrolls', 'core_pce', 'ffr', 'gs10']
ENDOG = HARD + ['vix']
MONTHLY = ['consumer_sentiment', 'epu', 'empire_state_mfg', 'philly_fed_mfg', 'dallas_fed_mfg']
ALL_SOFT = MONTHLY + ['sloos_ci_tightening']
TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]
SO = monthly_origins('2018-01-01', '2024-12-01')

signs = []


def dfm_prep_stvar(raw_ff, origin):
    sf = DFMSoftFactor(MONTHLY, quarterly_var='sloos_ci_tightening').fit(raw_ff)
    signs.append(sf.sign_)
    return pd.concat([raw_ff[ENDOG], sf.transform(raw_ff)], axis=1)


# ---- STVAR with DFM factor ----
stv = run_pseudo_real_time(
    lambda: STVAR(lags=4, transition_var='vix'),
    cfg, cubes, static, origins=SO, variables=ENDOG + ALL_SOFT,
    targets=TARGETS, horizons=HORIZONS, spec_name='hard_plus_dfm_factor',
    feature_prep=dfm_prep_stvar, progress=False,
)
stv.to_parquet('data/forecasts/short_window_stvar_dfmfactor.parquet')
print(f'STVAR DFM: {len(stv)} rows, ok {int(stv.ok.sum())}, skip {int((~stv.ok).sum())}')

# ---- MS-AR with DFM factor (univariate per target) ----
rows = []
for origin in SO:
    raw = build_feature_frame(cfg, cubes, static, origin, variables=TARGETS + ALL_SOFT)
    try:
        sf = DFMSoftFactor(MONTHLY, quarterly_var='sloos_ci_tightening').fit(raw)
        fcol = sf.transform(raw)
    except Exception:
        fcol = None
    for tgt in TARGETS:
        try:
            if fcol is None:
                raise ValueError('factor fit failed')
            ff = pd.concat([raw[[tgt]], fcol], axis=1)
            clean = ff.dropna(how='any')
            if len(clean) < 60:
                raise ValueError('too few obs')
            m = MSAR(order=2, k_regimes=2).fit(clean, [tgt, 'soft_f1'])
            f = m.forecast(max(HORIZONS))
            if not np.all(np.isfinite(f[tgt].values)):
                raise ValueError('nan forecast')
            for h in HORIZONS:
                fdate = f.index[h - 1]
                yhat = float(f.loc[fdate, tgt])
                yreal = get_realization(cfg, cubes, static, tgt, fdate, basis='latest')
                rows.append({'origin': origin, 'spec': 'hard_plus_dfm_factor', 'target': tgt,
                             'horizon': h, 'forecast': yhat, 'realized': yreal,
                             'error': yhat - yreal if pd.notna(yreal) else np.nan, 'ok': True})
        except Exception:
            for h in HORIZONS:
                rows.append({'origin': origin, 'spec': 'hard_plus_dfm_factor', 'target': tgt,
                             'horizon': h, 'forecast': np.nan, 'realized': np.nan,
                             'error': np.nan, 'ok': False})
msar = pd.DataFrame(rows)
msar.to_parquet('data/forecasts/short_window_msar_dfmfactor.parquet')
print(f'MS-AR DFM: {len(msar)} rows, ok {int(msar.ok.sum())}, skip {int((~msar.ok).sum())}')

s = pd.Series(signs)
print(f'\nfactor sign across {len(s)} STVAR refits: +1 in {int((s>0).sum())}, -1 in {int((s<0).sum())}'
      f'  ({"STABLE" if s.nunique()==1 else "FLIPS - investigate"})')
print('done')
