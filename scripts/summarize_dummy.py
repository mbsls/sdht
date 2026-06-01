import warnings; warnings.simplefilter('ignore')
import pandas as pd

van   = pd.read_parquet('data/forecasts/long_window_linear_var.parquet')
ld    = pd.read_parquet('data/forecasts/long_window_covid_dummy.parquet')
sd    = pd.read_parquet('data/forecasts/short_window_covid_dummy.parquet')


def rmse(res, spec):
    s = res[res.spec == spec].dropna(subset=['error']).copy()
    s['se'] = s['error'] ** 2
    return s.groupby(['target', 'horizon'])['se'].mean().pow(0.5)


# Long window: dummy vs vanilla, plus UMCSENT improvement within dummy
lo = pd.DataFrame({
    'vanilla_hard': rmse(van, 'hard_only'),
    'dummy_hard':   rmse(ld, 'hard_only'),
    'dummy_hardUM': rmse(ld, 'hard_plus_umcsent'),
})
lo['dummy_cuts_vs_vanilla_pct'] = 100 * (lo['vanilla_hard'] - lo['dummy_hard']) / lo['vanilla_hard']
lo['UM_imp_pct'] = 100 * (lo['dummy_hard'] - lo['dummy_hardUM']) / lo['dummy_hard']
lo.sort_index().round(3).to_csv('data/forecasts/_summary_long_dummy.csv')

# Short window: all three specs
sh = pd.DataFrame({
    'hard':     rmse(sd, 'hard_only'),
    'hardUM':   rmse(sd, 'hard_plus_umcsent'),
    'hardFac':  rmse(sd, 'hard_plus_soft_factor'),
})
sh['UM_imp_pct']  = 100 * (sh['hard'] - sh['hardUM']) / sh['hard']
sh['fac_imp_pct'] = 100 * (sh['hard'] - sh['hardFac']) / sh['hard']
sh.sort_index().round(3).to_csv('data/forecasts/_summary_short_dummy.csv')

print('wrote both summaries')
