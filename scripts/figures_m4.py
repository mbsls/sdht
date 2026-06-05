"""Paper figures for the M4 (STVAR) results.

Produces three figures in figs/:
  fig_m4_regime_helps_hard.png  -- finding 1: STVAR cuts hard-only RMSE
                                    vs the COVID-dummy linear VAR.
  fig_m4_soft_penalty.png       -- finding 2: adding soft data to the
                                    STVAR mostly hurts (DM significance).
  fig_m4_robustness_dummy.png   -- robustness (a): the soft penalty is
                                    the same or stronger without the
                                    COVID dummy.

All inputs load from data/forecasts/*.parquet + the cached VIX; no FRED.
"""
import warnings; warnings.simplefilter('ignore')
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from evaluation.gw import gw_test

FC = Path('data/forecasts')
vix_m = pickle.load(open('data/cache/inputs.pkl', 'rb'))['static']['VIXCLS']
vix_m.index = pd.to_datetime(vix_m.index); vix_m = vix_m.resample('MS').mean()

dummy_var   = pd.read_parquet(FC / 'long_window_covid_dummy.parquet')
stvar_long  = pd.read_parquet(FC / 'long_window_stvar.parquet')
stvar_short = pd.read_parquet(FC / 'short_window_stvar.parquet')
nod_long    = pd.read_parquet(FC / 'long_window_stvar_nodummy.parquet')
nod_short   = pd.read_parquet(FC / 'short_window_stvar_nodummy.parquet')

TARGETS = ['unemployment', 'industrial_production', 'payrolls', 'core_pce']
TLABEL = {'unemployment': 'Unemployment', 'industrial_production': 'Ind. production',
          'payrolls': 'Payrolls', 'core_pce': 'Core PCE'}
HORIZONS = [1, 3, 6, 12]


def rmse(res, spec):
    s = res[res.spec == spec].dropna(subset=['error']).copy()
    s['se'] = s['error'] ** 2
    return s.groupby(['target', 'horizon'])['se'].mean().pow(0.5)


def gw_rows(res, soft_spec):
    """Return dict[(target,horizon)] -> (soft_imp_pct, dm_p, better)."""
    out = {}
    for tgt in TARGETS:
        for h in HORIZONS:
            hard = res[(res.target == tgt) & (res.horizon == h) & (res.spec == 'hard_only')].set_index('origin')
            soft = res[(res.target == tgt) & (res.horizon == h) & (res.spec == soft_spec)].set_index('origin')
            idx = hard.index.intersection(soft.index)
            l1 = (hard.loc[idx, 'error'] ** 2); l2 = (soft.loc[idx, 'error'] ** 2)
            ok = l1.notna() & l2.notna()
            l1, l2 = l1[ok].values, l2[ok].values
            if len(l1) < 20:
                continue
            u = gw_test(l1, l2, tau=h, choice='uncond')
            imp = 100 * (np.sqrt(l1.mean()) - np.sqrt(l2.mean())) / np.sqrt(l1.mean())
            out[(tgt, h)] = (imp, u.pval, u.better_model)
    return out


# =====================================================================
# FIGURE 1 — finding 1: regime-switching helps hard data
# =====================================================================
dum_r = rmse(dummy_var, 'hard_only')
stv_r = rmse(stvar_long, 'hard_only')
imp = (100 * (dum_r - stv_r) / dum_r)

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(TARGETS))
bw = 0.2
cmap = plt.get_cmap('viridis')
for hi, h in enumerate(HORIZONS):
    vals = [imp.get((t, h), np.nan) for t in TARGETS]
    ax.bar(x + (hi - 1.5) * bw, vals, bw, color=cmap(hi / 3), label=f'h={h}')
ax.axhline(0, color='black', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([TLABEL[t] for t in TARGETS])
ax.set_ylabel('% RMSE reduction\n(STVAR vs COVID-dummy linear VAR)')
ax.set_title('Finding 1: regime-switching substantially improves HARD-data forecasts', fontsize=12)
ax.legend(title='Horizon', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig('figs/fig_m4_regime_helps_hard.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print('saved figs/fig_m4_regime_helps_hard.png')


# =====================================================================
# FIGURE 2 — finding 2: adding soft data to the STVAR mostly hurts
# =====================================================================
# Use the short-window soft factor (richest soft block) + long-window UMCSENT.
panels = [
    (stvar_long,  'hard_plus_umcsent',     'Long window: Hard + UMCSENT'),
    (stvar_short, 'hard_plus_soft_factor', 'Short window: Hard + soft factor'),
]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, (res, spec, title) in zip(axes, panels):
    g = gw_rows(res, spec)
    x = np.arange(len(TARGETS)); bw = 0.2
    for hi, h in enumerate(HORIZONS):
        vals, colors, hatches = [], [], []
        for t in TARGETS:
            imp_p = g.get((t, h))
            if imp_p is None:
                vals.append(np.nan); colors.append('white'); hatches.append('')
                continue
            imp_v, pval, better = imp_p
            vals.append(imp_v)
            # green = soft helps, red = soft hurts; dark if significant (p<0.10)
            if imp_v >= 0:
                colors.append('#1a9850' if pval < 0.10 else '#a6d96a')
            else:
                colors.append('#d73027' if pval < 0.10 else '#fdae61')
            hatches.append('')
        bars = ax.bar(x + (hi - 1.5) * bw, vals, bw, color=colors, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([TLABEL[t] for t in TARGETS], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
axes[0].set_ylabel('% RMSE change from adding soft data\n(+ = soft helps, − = soft hurts)')
legend_elems = [
    Patch(facecolor='#1a9850', edgecolor='black', label='soft helps (p<0.10)'),
    Patch(facecolor='#a6d96a', edgecolor='black', label='soft helps (n.s.)'),
    Patch(facecolor='#fdae61', edgecolor='black', label='soft hurts (n.s.)'),
    Patch(facecolor='#d73027', edgecolor='black', label='soft hurts (p<0.10)'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Finding 2: conditional on regime-switching, soft data mostly does not help (bars within each target: h=1,3,6,12)',
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig('figs/fig_m4_soft_penalty.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print('saved figs/fig_m4_soft_penalty.png')


# =====================================================================
# FIGURE 3 — robustness (a): dummy vs no-dummy soft penalty
# =====================================================================
# Focus on the short-window soft factor, the spec with the clearest signal.
g_dum = gw_rows(stvar_short, 'hard_plus_soft_factor')
g_nod = gw_rows(nod_short,   'hard_plus_soft_factor')

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, (g, title) in zip(axes, [(g_dum, 'With COVID dummy'), (g_nod, 'No COVID dummy (robustness a)')]):
    x = np.arange(len(TARGETS)); bw = 0.2
    for hi, h in enumerate(HORIZONS):
        vals, colors = [], []
        for t in TARGETS:
            ip = g.get((t, h))
            if ip is None:
                vals.append(np.nan); colors.append('white'); continue
            imp_v, pval, _ = ip
            vals.append(imp_v)
            if imp_v >= 0:
                colors.append('#1a9850' if pval < 0.10 else '#a6d96a')
            else:
                colors.append('#d73027' if pval < 0.10 else '#fdae61')
        ax.bar(x + (hi - 1.5) * bw, vals, bw, color=colors, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([TLABEL[t] for t in TARGETS], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
axes[0].set_ylabel('% RMSE change from soft factor\n(+ = helps, − = hurts)')
fig.legend(handles=legend_elems, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Robustness (a): the soft-data penalty is the same or stronger without the COVID dummy\n(short window, hard + soft factor; bars h=1,3,6,12)',
             fontsize=12, y=1.04)
plt.tight_layout()
fig.savefig('figs/fig_m4_robustness_dummy.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print('saved figs/fig_m4_robustness_dummy.png')
