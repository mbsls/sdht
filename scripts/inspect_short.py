import warnings; warnings.simplefilter('ignore')
import pandas as pd

sd = pd.read_parquet('data/forecasts/short_window_covid_dummy.parquet')
print('rows:', len(sd), 'ok:', int(sd.ok.sum()))
print('specs:', sorted(sd.spec.unique()))
print('targets:', sorted(sd.target.unique()))

for tgt in ['unemployment', 'industrial_production']:
    for spec in ['hard_only', 'hard_plus_umcsent', 'hard_plus_soft_factor']:
        s = sd[(sd.target == tgt) & (sd.horizon == 3) & (sd.spec == spec)].dropna(subset=['error'])
        rmse = (s.error ** 2).mean() ** 0.5 if len(s) else float('nan')
        print(f'{tgt:24s} h=3 {spec:24s} n={len(s):3d} rmse={rmse:.3f}')
