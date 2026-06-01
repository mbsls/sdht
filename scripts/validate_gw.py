"""Validate the GW port: (1) unconditional reduces to DM t-stat^2,
(2) detects a regime-dependent loss differential the unconditional test misses."""
import warnings; warnings.simplefilter('ignore')
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import stats
from evaluation.gw import gw_test, dm_test

rng = np.random.RandomState(0)
T = 400

# ---- Case 1: equal predictive ability -> should NOT reject ----
e1 = rng.randn(T)
e2 = rng.randn(T)
l1, l2 = e1**2, e2**2
r = dm_test(l1, l2, tau=1)
print(f"[equal] uncond GW: stat={r.tstat:.3f} p={r.pval:.3f}  (expect p large, no rejection)")

# Compare to a hand-rolled DM t-stat (tau=1): t = mean(d)/se(mean(d)), chi2 = t^2
d = l1 - l2
dm_t = d.mean() / (d.std(ddof=1) / np.sqrt(T))
print(f"        hand DM t^2 = {dm_t**2:.3f}  vs GW stat {r.tstat:.3f}  (should match closely)")

# ---- Case 2: model 2 uniformly better -> unconditional rejects ----
l2b = (0.5 * e2)**2     # model 2 has much smaller errors
r2 = dm_test(l1, l2b, tau=1)
print(f"\n[m2 better] uncond GW: stat={r2.tstat:.3f} p={r2.pval:.4f} better=model{r2.better_model}  (expect reject, model2)")

# ---- Case 3: regime-dependent advantage that nets to ZERO unconditionally ----
# In high-regime periods model 2 is better; in low-regime model 1 is better;
# on average the loss differential is ~0, so unconditional DM should miss it,
# but conditional GW with the regime instrument should detect it.
regime = (rng.rand(T) > 0.5).astype(float)          # 0/1 regime known at origin
adv = np.where(regime == 1, -1.0, +1.0)             # loss diff sign flips by regime
d3 = adv + 0.3 * rng.randn(T)                        # noisy
# Construct losses with this differential
base = rng.randn(T)**2
l1_3 = base + np.maximum(d3, 0)
l2_3 = base + np.maximum(-d3, 0)

r_uncond = gw_test(l1_3, l2_3, tau=1, choice="uncond")
r_cond   = gw_test(l1_3, l2_3, tau=1, choice="cond", instruments=regime)
print(f"\n[regime-dependent, nets to ~0]")
print(f"   mean loss diff = {(l1_3-l2_3).mean():+.3f}")
print(f"   uncond GW: stat={r_uncond.tstat:6.3f} p={r_uncond.pval:.4f}  (expect MISS -> p large)")
print(f"   cond  GW: stat={r_cond.tstat:6.3f} p={r_cond.pval:.4f}  (expect DETECT -> p small)")

# ---- Case 4: multi-step HAC sanity (tau=6 shouldn't crash, dof correct) ----
r4 = gw_test(l1_3, l2_3, tau=6, choice="cond", instruments=regime)
print(f"\n[tau=6 HAC] stat={r4.tstat:.3f} p={r4.pval:.4f} dof={r4.dof} n={r4.n}")
print("\nvalidation done")
