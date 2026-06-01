import warnings; warnings.simplefilter('ignore')
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from fredapi import Fred
from load_config import load_config
from evaluation.feature_frame import load_inputs

cfg = load_config('configs/us.yaml')
fred = Fred(api_key=cfg['api_key'])

# FRED rate limits aggressively; retry with backoff (Python sleep is fine).
for attempt in range(1, 11):
    try:
        cubes, static = load_inputs(cfg, fred)
        break
    except ValueError as e:
        if 'Too Many Requests' not in str(e) and 'Rate Limit' not in str(e):
            raise
        wait = min(60, 10 * attempt)
        print(f'attempt {attempt}: rate-limited, sleeping {wait}s', flush=True)
        time.sleep(wait)
else:
    raise SystemExit('still rate-limited after 10 attempts')

Path('data/cache').mkdir(parents=True, exist_ok=True)
with open('data/cache/inputs.pkl', 'wb') as f:
    pickle.dump({'cubes': cubes, 'static': static}, f)
print(f'cached {len(cubes)} cubes + {len(static)} static series')
