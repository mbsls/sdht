"""Compare MS-AR results: GW panels + calm/stress decomposition.

Answers, for the latent-regime model:
1. No explosions? (gate before any RMSE)
2. Does soft data help? GW panel, both windows.
3. Is any soft effect concentrated in stress origins, or calm-driven?
   (the decomposition the user flagged as central)
"""
import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from evaluation.gw import gw_test

vix_m = pickle.load(open('data/cache/inputs.pkl', 'rb'))['static']['VIXCLS']
vix_m.index = pd.to_datetime(vix_m.index); vix_m = vix_m.resample('MS').mean()

ms_long = pd.read_parquet('data/forecasts/long_window_msar.parquet')
ms_short = pd.read_parquet('data/forecasts/short_window_msar.parquet')

TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]


def gate(res, label):
    ok = res[res.ok].copy(); ok['absf'] = ok['forecast'].abs()
    print(f'  [{label}] ok {int(res.ok.sum())}/{len(res)}, skip {int((~res.ok).sum())}, '
          f'max|forecast| {ok.absf.max():.2f}')


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


def decomp(res, soft_spec, label, thr=25):
    print(f'\n=== calm vs stress decomposition: {label} (VIX threshold {thr}) ===')
    for tgt in TARGETS:
        for h in [3, 6]:
            hard = res[(res.target == tgt) & (res.horizon == h) & (res.spec == 'hard_only')].set_index('origin')
            soft = res[(res.target == tgt) & (res.horizon == h) & (res.spec == soft_spec)].set_index('origin')
            idx = hard.index.intersection(soft.index)
            d = pd.DataFrame({'lh': hard.loc[idx, 'error'] ** 2, 'ls': soft.loc[idx, 'error'] ** 2,
                              'vix': vix_m.reindex(idx)}).dropna()
            calm, stress = d[d.vix < thr], d[d.vix >= thr]

            def imp(x):
                return (100 * (np.sqrt(x.lh.mean()) - np.sqrt(x.ls.mean())) / np.sqrt(x.lh.mean())
                        if len(x) else np.nan)
            print(f'  {tgt:22} h={h}: calm n={len(calm):3d} imp={imp(calm):+6.1f}% | '
                  f'stress n={len(stress):2d} imp={imp(stress):+6.1f}%')


print('=== explosion gate ===')
gate(ms_long, 'MS-AR long')
gate(ms_short, 'MS-AR short')

print('\n=== GW: MS-AR long, hard vs hard+UMCSENT ===')
g1 = gw_panel(ms_long, 'hard_plus_umcsent'); print(g1.round(4).to_string(index=False))
g1.round(4).to_csv('data/forecasts/_gw_msar_long_UMCSENT.csv', index=False)

print('\n=== GW: MS-AR short, hard vs hard+soft factor ===')
g2 = gw_panel(ms_short, 'hard_plus_soft_factor'); print(g2.round(4).to_string(index=False))
g2.round(4).to_csv('data/forecasts/_gw_msar_short_FACTOR.csv', index=False)

decomp(ms_long, 'hard_plus_umcsent', 'MS-AR long, UMCSENT')
decomp(ms_short, 'hard_plus_soft_factor', 'MS-AR short, soft factor')
print('\ndone')
