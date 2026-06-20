"""Compare the DFM (mixed-frequency, +SLOOS) soft factor against the
static-PCA soft factor, for both STVAR and MS-AR, short window.

Key question: does adding SLOOS via a proper state-space mixed-frequency
factor change the soft-vs-hard verdict -- especially the one positive
result (STVAR/MS-AR industrial production)?
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

TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
HORIZONS = [1, 3, 6, 12]

DATA = {
    'STVAR': {
        'pca': ('data/forecasts/short_window_stvar.parquet', 'hard_plus_soft_factor'),
        'dfm': ('data/forecasts/short_window_stvar_dfmfactor.parquet', 'hard_plus_dfm_factor'),
        'hard_src': 'data/forecasts/short_window_stvar.parquet',
    },
    'MS-AR': {
        'pca': ('data/forecasts/short_window_msar.parquet', 'hard_plus_soft_factor'),
        'dfm': ('data/forecasts/short_window_msar_dfmfactor.parquet', 'hard_plus_dfm_factor'),
        'hard_src': 'data/forecasts/short_window_msar.parquet',
    },
}


def gw_vs_hard(soft_path, soft_spec, hard_path):
    soft_res = pd.read_parquet(soft_path)
    hard_res = pd.read_parquet(hard_path)
    rows = []
    for tgt in TARGETS:
        for h in HORIZONS:
            hard = hard_res[(hard_res.target == tgt) & (hard_res.horizon == h) & (hard_res.spec == 'hard_only')].set_index('origin')
            soft = soft_res[(soft_res.target == tgt) & (soft_res.horizon == h) & (soft_res.spec == soft_spec)].set_index('origin')
            idx = hard.index.intersection(soft.index)
            l1 = (hard.loc[idx, 'error'] ** 2); l2 = (soft.loc[idx, 'error'] ** 2)
            vx = vix_m.reindex(idx)
            ok = l1.notna() & l2.notna() & vx.notna()
            l1, l2, vx = l1[ok].values, l2[ok].values, vx[ok].values
            if len(l1) < 20:
                continue
            u = gw_test(l1, l2, tau=h, choice='uncond')
            c = gw_test(l1, l2, tau=h, choice='cond', instruments=vx)
            calm = vx < 25; stress = vx >= 25
            def imp(mask):
                if mask.sum() < 3: return np.nan
                return 100 * (np.sqrt(l1[mask].mean()) - np.sqrt(l2[mask].mean())) / np.sqrt(l1[mask].mean())
            rows.append({'target': tgt, 'h': h, 'n': len(l1),
                         'imp_%': round(100 * (np.sqrt(l1.mean()) - np.sqrt(l2.mean())) / np.sqrt(l1.mean()), 2),
                         'DM_p': round(u.pval, 4),
                         'imp_calm': round(imp(calm), 1), 'imp_stress': round(imp(stress), 1),
                         'n_stress': int(stress.sum())})
    return pd.DataFrame(rows)


for model, d in DATA.items():
    print(f'\n{"="*70}\n{model}: PCA factor vs DFM factor (+SLOOS), short window\n{"="*70}')
    for tag in ['pca', 'dfm']:
        path, spec = d[tag]
        if not Path(path).exists():
            print(f'  [{tag}] missing {path}'); continue
        g = gw_vs_hard(path, spec, d['hard_src'])
        print(f'\n  --- {tag.upper()} ---')
        print(g.to_string(index=False))
print('\ndone')
