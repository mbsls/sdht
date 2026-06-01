import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import pandas as pd

from load_config import load_config
from evaluation.harness import run_pseudo_real_time, monthly_origins
from models.covid_dummy_var import CovidDummyVAR
from models.soft_factor import SoftFactor

cfg = load_config('configs/us.yaml')
with open('data/cache/inputs.pkl', 'rb') as f:
    d = pickle.load(f)
cubes, static = d['cubes'], d['static']

HARD = ['unemployment', 'industrial_production', 'payrolls', 'core_pce', 'ffr', 'gs10']
SOFT_FAC = ['consumer_sentiment', 'epu', 'empire_state_mfg', 'philly_fed_mfg', 'dallas_fed_mfg']
TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]


def mk():
    return CovidDummyVAR(lags=4, window=('2020-03-01', '2021-06-01'))


def factor_prep(hard, soft):
    def prep(raw_ff, origin):
        sf = SoftFactor(soft, n_components=1, sign_reference='consumer_sentiment').fit(raw_ff)
        return pd.concat([raw_ff[hard], sf.transform(raw_ff)], axis=1)
    return prep


# ---- long window ----
LO = monthly_origins('2001-01-01', '2024-12-01')
l1 = run_pseudo_real_time(mk, cfg, cubes, static, origins=LO, variables=HARD,
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_only', progress=False)
l2 = run_pseudo_real_time(mk, cfg, cubes, static, origins=LO, variables=HARD + ['consumer_sentiment'],
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_plus_umcsent', progress=False)
pd.concat([l1, l2], ignore_index=True).to_parquet('data/forecasts/long_window_covid_dummy.parquet')

# ---- short window ----
SO = monthly_origins('2018-01-01', '2024-12-01')
s1 = run_pseudo_real_time(mk, cfg, cubes, static, origins=SO, variables=HARD,
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_only', progress=False)
s2 = run_pseudo_real_time(mk, cfg, cubes, static, origins=SO, variables=HARD + ['consumer_sentiment'],
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_plus_umcsent', progress=False)
s3 = run_pseudo_real_time(mk, cfg, cubes, static, origins=SO, variables=HARD + SOFT_FAC,
                          targets=TARGETS, horizons=HORIZONS, spec_name='hard_plus_soft_factor',
                          feature_prep=factor_prep(HARD, SOFT_FAC), progress=False)
pd.concat([s1, s2, s3], ignore_index=True).to_parquet('data/forecasts/short_window_covid_dummy.parquet')

print('done both windows')
