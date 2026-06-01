"""Run STVAR pseudo-real-time forecasts on both windows.

VIX is part of the endogenous block (it is the transition variable and
is forecast by the system). Specs differ only by which *soft* block, if
any, is added on top of the hard block.
"""
import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from load_config import load_config
from evaluation.harness import run_pseudo_real_time, monthly_origins
from models.stvar import STVAR
from models.soft_factor import SoftFactor

cfg = load_config('configs/us.yaml')
cubes, static = pickle.load(open('data/cache/inputs.pkl', 'rb')).values()

HARD = ['unemployment', 'industrial_production', 'payrolls', 'core_pce', 'ffr', 'gs10']
ENDOG = HARD + ['vix']                       # vix always in the system
SOFT_FAC = ['consumer_sentiment', 'epu', 'empire_state_mfg', 'philly_fed_mfg', 'dallas_fed_mfg']
TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]

# track the chosen (gamma, c) to detect corner solutions
chosen = []


def mk():
    m = STVAR(lags=4, transition_var='vix')
    return m


def mk_tracked():
    """Factory that records the fitted (gamma,c) via a thin wrapper."""
    base = STVAR(lags=4, transition_var='vix')
    orig_fit = base.fit

    def fit(df, variables):
        r = orig_fit(df, variables)
        chosen.append((r.gamma_, r.c_))
        return r
    base.fit = fit
    return base


def factor_prep(hard_plus_vix, soft):
    def prep(raw_ff, origin):
        sf = SoftFactor(soft, n_components=1, sign_reference='consumer_sentiment').fit(raw_ff)
        return pd.concat([raw_ff[hard_plus_vix], sf.transform(raw_ff)], axis=1)
    return prep


# ---- long window ----
LO = monthly_origins('2001-01-01', '2024-12-01')
l1 = run_pseudo_real_time(mk_tracked, cfg, cubes, static, origins=LO, variables=ENDOG,
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_only', progress=False)
l2 = run_pseudo_real_time(mk, cfg, cubes, static, origins=LO, variables=ENDOG + ['consumer_sentiment'],
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_plus_umcsent', progress=False)
pd.concat([l1, l2], ignore_index=True).to_parquet('data/forecasts/long_window_stvar.parquet')

# ---- short window ----
SO = monthly_origins('2018-01-01', '2024-12-01')
s1 = run_pseudo_real_time(mk, cfg, cubes, static, origins=SO, variables=ENDOG,
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_only', progress=False)
s2 = run_pseudo_real_time(mk, cfg, cubes, static, origins=SO, variables=ENDOG + ['consumer_sentiment'],
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_plus_umcsent', progress=False)
s3 = run_pseudo_real_time(mk, cfg, cubes, static, origins=SO, variables=ENDOG + SOFT_FAC,
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_plus_soft_factor',
                          feature_prep=factor_prep(ENDOG, SOFT_FAC), progress=False)
pd.concat([s1, s2, s3], ignore_index=True).to_parquet('data/forecasts/short_window_stvar.parquet')

# ---- report grid choices (corner check) ----
ch = pd.DataFrame(chosen, columns=['gamma', 'c'])
print('STVAR grid choices over long-window hard_only origins:')
print('  gamma counts:', ch['gamma'].value_counts().sort_index().to_dict())
print('  c counts:    ', ch['c'].value_counts().sort_index().to_dict())
print('done both windows')
