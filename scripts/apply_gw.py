"""Apply the GW test to the COVID-dummy forecast results.

For each (target, horizon), compare hard-only vs a soft spec using:
  - unconditional GW (= Diebold-Mariano)
  - conditional GW with the VIX level at the forecast origin as instrument

A small conditional p-value means the relative performance of soft vs
hard depends on the VIX regime -- the paper's central question.
"""
import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from evaluation.gw import gw_test

with open('data/cache/inputs.pkl', 'rb') as f:
    static = pickle.load(f)['static']
vix = static['VIXCLS']; vix.index = pd.to_datetime(vix.index)
vix_m = vix.resample('MS').mean()

LONG  = pd.read_parquet('data/forecasts/long_window_covid_dummy.parquet')
SHORT = pd.read_parquet('data/forecasts/short_window_covid_dummy.parquet')

TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]


def panel(res, soft_spec, label):
    rows = []
    for tgt in TARGETS:
        for h in HORIZONS:
            hard = res[(res.target == tgt) & (res.horizon == h) & (res.spec == 'hard_only')].set_index('origin')
            soft = res[(res.target == tgt) & (res.horizon == h) & (res.spec == soft_spec)].set_index('origin')
            idx = hard.index.intersection(soft.index)
            l1 = (hard.loc[idx, 'error'] ** 2)        # hard loss
            l2 = (soft.loc[idx, 'error'] ** 2)        # soft loss
            vx = vix_m.reindex(idx)

            valid = l1.notna() & l2.notna() & vx.notna()
            l1v, l2v, vxv = l1[valid].values, l2[valid].values, vx[valid].values
            if len(l1v) < 20:
                continue

            uncond = gw_test(l1v, l2v, tau=h, choice='uncond')
            cond   = gw_test(l1v, l2v, tau=h, choice='cond', instruments=vxv)

            rmse_hard = np.sqrt(np.mean(l1v))
            rmse_soft = np.sqrt(np.mean(l2v))
            rows.append({
                'target': tgt, 'horizon': h, 'n': len(l1v),
                'rmse_hard': rmse_hard, 'rmse_soft': rmse_soft,
                'soft_imp_%': 100 * (rmse_hard - rmse_soft) / rmse_hard,
                'DM_p': uncond.pval,
                'GW_vix_p': cond.pval,
                'better': f'soft' if uncond.better_model == 2 else 'hard',
            })
    return pd.DataFrame(rows)


out = {}
out['long_UMCSENT']  = panel(LONG,  'hard_plus_umcsent',     'long: hard vs hard+UMCSENT')
out['short_UMCSENT'] = panel(SHORT, 'hard_plus_umcsent',     'short: hard vs hard+UMCSENT')
out['short_FACTOR']  = panel(SHORT, 'hard_plus_soft_factor', 'short: hard vs hard+soft factor')

for k, v in out.items():
    v.round(4).to_csv(f'data/forecasts/_gw_{k}.csv', index=False)

print('wrote GW result tables')
for k, v in out.items():
    print(f'\n===== {k} =====')
    print(v.round(4).to_string(index=False))
