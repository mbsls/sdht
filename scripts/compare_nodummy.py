"""Compare no-dummy STVAR to dummy STVAR.

Two questions:
1. Does removing the COVID dummy change the HARD-only forecast quality?
2. Does finding 2 (soft adds little / hurts) survive without the dummy?
   -> GW panels, soft vs hard, both windows.

Also verifies no explosions (max|forecast|) before reporting.
"""
import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from evaluation.gw import gw_test

vix_m = pickle.load(open('data/cache/inputs.pkl', 'rb'))['static']['VIXCLS']
vix_m.index = pd.to_datetime(vix_m.index)
vix_m = vix_m.resample('MS').mean()

dum_long = pd.read_parquet('data/forecasts/long_window_stvar.parquet')
nod_long = pd.read_parquet('data/forecasts/long_window_stvar_nodummy.parquet')
nod_short = pd.read_parquet('data/forecasts/short_window_stvar_nodummy.parquet')

TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]


def explosion_check(res, label):
    ok = res[res.ok].copy(); ok['absf'] = ok['forecast'].abs()
    print(f'  [{label}] rows {len(res)}, ok {int(res.ok.sum())}, skip {int((~res.ok).sum())}, '
          f'max|forecast| {ok.absf.max():.2f}')


def rmse(res, spec):
    s = res[res.spec == spec].dropna(subset=['error']).copy()
    s['se'] = s['error'] ** 2
    return s.groupby(['target', 'horizon'])['se'].mean().pow(0.5)


def gw_panel(res, soft_spec):
    rows = []
    for tgt in TARGETS:
        for h in HORIZONS:
            hard = res[(res.target == tgt) & (res.horizon == h) & (res.spec == 'hard_only')].set_index('origin')
            soft = res[(res.target == tgt) & (res.horizon == h) & (res.spec == soft_spec)].set_index('origin')
            idx = hard.index.intersection(soft.index)
            l1 = (hard.loc[idx, 'error'] ** 2); l2 = (soft.loc[idx, 'error'] ** 2)
            vx = vix_m.reindex(idx)
            ok = l1.notna() & l2.notna() & vx.notna()
            l1, l2, vx = l1[ok].values, l2[ok].values, vx[ok].values
            if len(l1) < 20:
                continue
            u = gw_test(l1, l2, tau=h, choice='uncond')
            c = gw_test(l1, l2, tau=h, choice='cond', instruments=vx)
            rows.append({'target': tgt, 'horizon': h, 'n': len(l1),
                         'soft_imp_%': 100 * (np.sqrt(l1.mean()) - np.sqrt(l2.mean())) / np.sqrt(l1.mean()),
                         'DM_p': u.pval, 'GW_vix_p': c.pval,
                         'better': 'soft' if u.better_model == 2 else 'hard'})
    return pd.DataFrame(rows)


print('=== explosion check ===')
explosion_check(nod_long, 'nodummy long')
explosion_check(nod_short, 'nodummy short')

print('\n=== Q1: no-dummy vs dummy STVAR (hard-only, long window) ===')
q1 = pd.DataFrame({'dummy': rmse(dum_long, 'hard_only'),
                   'nodummy': rmse(nod_long, 'hard_only')})
q1['nodummy_better_%'] = 100 * (q1['dummy'] - q1['nodummy']) / q1['dummy']
print(q1.round(3).to_string())

print('\n=== Q2a: no-dummy GW, long, hard vs hard+UMCSENT ===')
g_long = gw_panel(nod_long, 'hard_plus_umcsent')
print(g_long.round(4).to_string(index=False))
g_long.round(4).to_csv('data/forecasts/_gw_nodummy_long_UMCSENT.csv', index=False)

print('\n=== Q2b: no-dummy GW, short, hard vs hard+soft factor ===')
g_short = gw_panel(nod_short, 'hard_plus_soft_factor')
print(g_short.round(4).to_string(index=False))
g_short.round(4).to_csv('data/forecasts/_gw_nodummy_short_FACTOR.csv', index=False)

print('\ndone')
