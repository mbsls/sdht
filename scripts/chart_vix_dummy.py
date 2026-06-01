import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('data/cache/inputs.pkl', 'rb') as f:
    static = pickle.load(f)['static']

vix = static['VIXCLS']
vix.index = pd.to_datetime(vix.index)
vix_m = vix.resample('MS').mean()

res_long  = pd.read_parquet('data/forecasts/long_window_covid_dummy.parquet')
res_short = pd.read_parquet('data/forecasts/short_window_covid_dummy.parquet')

COL = {'hard_only': '#1f77b4', 'hard_plus_umcsent': '#2ca02c', 'hard_plus_soft_factor': '#d62728'}
LAB = {'hard_only': 'Hard only', 'hard_plus_umcsent': 'Hard + UMCSENT', 'hard_plus_soft_factor': 'Hard + soft factor'}


def rmse_by_vix_bin(res, target, horizon):
    m = res[res['ok']].copy()
    m = m.merge(vix_m.rename('vix_origin'), left_on='origin', right_index=True, how='left')
    m['vix_bin'] = pd.cut(m['vix_origin'], bins=[0, 15, 20, 25, 35, 200],
                          labels=['<15', '15-20', '20-25', '25-35', '>=35'])
    sub = m[(m['target'] == target) & (m['horizon'] == horizon)].copy()
    sub['se'] = sub['error'] ** 2
    rmse = sub.groupby(['vix_bin', 'spec'], observed=True)['se'].mean().pow(0.5).unstack('spec')
    n = sub.groupby('vix_bin', observed=True).size() // sub['spec'].nunique()
    return rmse, n


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
panels = [
    ('unemployment', 6, res_long, 'Unemployment, h=6 (long window)'),
    ('industrial_production', 6, res_long, 'Industrial production, h=6 (long window)'),
    ('unemployment', 6, res_short, 'Unemployment, h=6 (short window, 3 specs)'),
    ('industrial_production', 6, res_short, 'Industrial production, h=6 (short window, 3 specs)'),
]
for ax, (tgt, h, res, title) in zip(axes.flatten(), panels):
    rmse, n = rmse_by_vix_bin(res, tgt, h)
    specs = [c for c in ['hard_only', 'hard_plus_umcsent', 'hard_plus_soft_factor'] if c in rmse.columns]
    bins = list(rmse.index)
    x = np.arange(len(bins))
    bw = 0.8 / len(specs)
    for si, spec in enumerate(specs):
        ax.bar(x + (si - (len(specs) - 1) / 2) * bw, rmse[spec].values, bw,
               color=COL[spec], alpha=0.85, label=LAB[spec], edgecolor='black', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'VIX {b}\n(n={int(n.loc[b])})' for b in bins], fontsize=9)
    ax.set_ylabel('RMSE')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('RMSE by VIX-at-origin regime — COVID modeled via dummy + lag interaction (full sample)',
             y=0.995, fontsize=12)
plt.tight_layout()
fig.savefig('figs/chart_vix_conditional_dummy.png', dpi=110, bbox_inches='tight')
print('saved figs/chart_vix_conditional_dummy.png')
