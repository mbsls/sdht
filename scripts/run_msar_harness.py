"""Robustness (b): univariate Markov-switching AR harness.

Unlike the VAR specs, MS-AR is univariate per target. "Soft data" enters
as an exogenous regressor with a common (non-switching) coefficient, so
the comparison is a clean nested test: AR(p) with regime-switching mean/
variance, vs the same plus a soft regressor.

  hard_only             : MS-AR(order) on the target alone
  hard_plus_umcsent     : + Michigan sentiment as exog
  hard_plus_soft_factor : + the recursively-refit soft PCA factor as exog
                          (short window only)

The latent regime needs no VIX threshold, so this is a different
identification of "hard times" than the STVAR. Writes *_msar.parquet.
"""
import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from load_config import load_config
from evaluation.feature_frame import build_feature_frame, get_realization
from evaluation.harness import monthly_origins
from models.msar import MSAR
from models.soft_factor import SoftFactor

cfg = load_config('configs/us.yaml')
cubes, static = pickle.load(open('data/cache/inputs.pkl', 'rb')).values()

TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]
SOFT_FAC = ['consumer_sentiment', 'epu', 'empire_state_mfg', 'philly_fed_mfg', 'dallas_fed_mfg']


def run(origins, specs, out_path, label):
    """specs: list of (spec_name, builder) where builder(origin, target)
    returns (feature_df, var_list) with var_list[0]=target."""
    rows = []
    for origin in origins:
        for spec_name, builder in specs:
            for tgt in TARGETS:
                try:
                    ff, vlist = builder(origin, tgt)
                    clean = ff[vlist].dropna(how='any')
                    if len(clean) < 60:
                        raise ValueError('too few obs')
                    m = MSAR(order=2, k_regimes=2).fit(clean, vlist)
                    f = m.forecast(max(HORIZONS))
                    if not np.all(np.isfinite(f[tgt].values)):
                        raise ValueError('nan forecast')
                    for h in HORIZONS:
                        fdate = f.index[h - 1]
                        yhat = float(f.loc[fdate, tgt])
                        yreal = get_realization(cfg, cubes, static, tgt, fdate, basis='latest')
                        rows.append({'origin': origin, 'spec': spec_name, 'target': tgt,
                                     'horizon': h, 'forecast': yhat, 'realized': yreal,
                                     'error': yhat - yreal if pd.notna(yreal) else np.nan,
                                     'ok': True})
                except Exception:
                    for h in HORIZONS:
                        rows.append({'origin': origin, 'spec': spec_name, 'target': tgt,
                                     'horizon': h, 'forecast': np.nan, 'realized': np.nan,
                                     'error': np.nan, 'ok': False})
    res = pd.DataFrame(rows)
    res.to_parquet(out_path)
    print(f'{label}: {len(res)} rows, ok {int(res.ok.sum())}, skip {int((~res.ok).sum())}')
    return res


def b_hard(origin, tgt):
    ff = build_feature_frame(cfg, cubes, static, origin, variables=[tgt])
    return ff, [tgt]


def b_umcsent(origin, tgt):
    ff = build_feature_frame(cfg, cubes, static, origin, variables=[tgt, 'consumer_sentiment'])
    return ff, [tgt, 'consumer_sentiment']


def b_factor(origin, tgt):
    raw = build_feature_frame(cfg, cubes, static, origin, variables=[tgt] + SOFT_FAC)
    sf = SoftFactor(SOFT_FAC, n_components=1, sign_reference='consumer_sentiment').fit(raw)
    ff = pd.concat([raw[[tgt]], sf.transform(raw)], axis=1)
    return ff, [tgt, 'soft_f1']


# ---- long window ----
LO = monthly_origins('2001-01-01', '2024-12-01')
run(LO, [('hard_only', b_hard), ('hard_plus_umcsent', b_umcsent)],
    'data/forecasts/long_window_msar.parquet', 'long')

# ---- short window ----
SO = monthly_origins('2018-01-01', '2024-12-01')
run(SO, [('hard_only', b_hard), ('hard_plus_umcsent', b_umcsent),
         ('hard_plus_soft_factor', b_factor)],
    'data/forecasts/short_window_msar.parquet', 'short')

print('done both windows (MS-AR)')
