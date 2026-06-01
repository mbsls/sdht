import warnings; warnings.simplefilter('ignore')
import pandas as pd

print("=== files ===")
import os
for f in sorted(os.listdir('data/forecasts')):
    if f.endswith('.parquet'):
        print(f, os.path.getsize(os.path.join('data/forecasts', f)))

print("\n=== short-window dummy: per (target,horizon,spec) rmse ===")
sd = pd.read_parquet('data/forecasts/short_window_covid_dummy.parquet')
print("rows", len(sd), "specs", sorted(sd.spec.unique()))
s = sd.dropna(subset=['error']).copy()
s['se'] = s['error'] ** 2
tab = s.groupby(['target', 'horizon', 'spec'])['se'].mean().pow(0.5).round(3)
print(tab.to_string())
