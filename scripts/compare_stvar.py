"""Compare STVAR vs COVID-dummy VAR, and run GW tests on STVAR results.

Three questions:
1. Does STVAR forecast better than the dummy VAR (hard-only)? RMSE ratio.
2. Within STVAR, does adding soft data help? (hard vs hard+soft)
3. Is the soft edge concentrated in high-VIX regimes? GW-conditional-on-VIX.
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

dummy_long  = pd.read_parquet('data/forecasts/long_window_covid_dummy.parquet')
stvar_long  = pd.read_parquet('data/forecasts/long_window_stvar.parquet')
stvar_short = pd.read_parquet('data/forecasts/short_window_stvar.parquet')

TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]


def rmse(res, spec):
    s = res[res.spec == spec].dropna(subset=['error']).copy()
    s['se'] = s['error'] ** 2
    return s.groupby(['target', 'horizon'])['se'].mean().pow(0.5)


# Q1: STVAR vs dummy (hard-only), long window
q1 = pd.DataFrame({'dummy_hard': rmse(dummy_long, 'hard_only'),
                   'stvar_hard': rmse(stvar_long, 'hard_only')})
q1['stvar_better_%'] = 100 * (q1['dummy_hard'] - q1['stvar_hard']) / q1['dummy_hard']
q1.round(3).to_csv('data/forecasts/_stvar_vs_dummy.csv')


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


gw_long_um  = gw_panel(stvar_long, 'hard_plus_umcsent')
gw_short_um = gw_panel(stvar_short, 'hard_plus_umcsent')
gw_short_fac = gw_panel(stvar_short, 'hard_plus_soft_factor')
gw_long_um.round(4).to_csv('data/forecasts/_gw_stvar_long_UMCSENT.csv', index=False)
gw_short_um.round(4).to_csv('data/forecasts/_gw_stvar_short_UMCSENT.csv', index=False)
gw_short_fac.round(4).to_csv('data/forecasts/_gw_stvar_short_FACTOR.csv', index=False)

print('=== Q1: STVAR vs dummy VAR (hard-only, long window) ===')
print(q1.round(3).to_string())
print('\n=== Q3a: GW, STVAR long, hard vs hard+UMCSENT ===')
print(gw_long_um.round(4).to_string(index=False))
print('\n=== Q3b: GW, STVAR short, hard vs hard+soft factor ===')
print(gw_short_fac.round(4).to_string(index=False))
print('\nwrote comparison CSVs')
