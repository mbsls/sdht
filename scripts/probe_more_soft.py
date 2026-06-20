"""Probe FRED/ALFRED for additional soft-data surveys with real-time vintages.

Reports, per candidate: obs range, vintage range + count (ALFRED), freq.
A series is only usable for the real-time exercise if it has vintages.
"""
import sys, time
import yaml
from fredapi import Fred

cfg = yaml.safe_load(open('config.yaml'))
fred = Fred(api_key=cfg['api_key'])

CANDIDATES = {
    # business surveys
    'NFIB Small Business Optimism': 'USACSCICP02STSAM',   # OECD BCI proxy; check
    'ISM Mfg PMI (manufacturing)': 'MANEMP',              # placeholder sanity (not PMI)
    'ISM Mfg (NAPM legacy)': 'NAPM',
    'ISM Services (NMFCI legacy)': 'NAPMNMI',
    # regional Fed - services / other districts
    'Richmond Fed Mfg composite': 'RCMFGCI',
    'KC Fed Mfg composite': 'KCFMFG',
    'NY Fed services activity': 'GACDISA066MSFRBNY',      # already have mfg; services?
    'Philly Fed nonmfg': 'GANNFSA156MSFRBPHI',
    'Dallas Fed services': 'TSSOSAMFRBDAL',
    # consumer
    'Conf Board Consumer Confidence': 'CONCCONF',
    'Michigan current conditions': 'UMCSENT1',
    'Michigan expectations': 'UMCSENT2',
    # uncertainty / financial sentiment
    'OECD US Consumer Conf': 'CSCICP03USM665S',
    'OECD US Business Conf': 'BSCICP03USM665S',
}


def probe(code):
    try:
        s = fred.get_series(code)
    except Exception as e:
        return ('MISSING', str(e).splitlines()[0][:50])
    try:
        v = fred.get_series_vintage_dates(code)
        vinfo = f"vint {min(v).date()}..{max(v).date()} n={len(v)}"
    except Exception:
        vinfo = "NO VINTAGES"
    freq = s.index.inferred_freq or '?'
    return ('OK', f"obs {s.index.min().date()}..{s.index.max().date()} freq~{freq} | {vinfo}")


for label, code in CANDIDATES.items():
    status, info = probe(code)
    print(f"[{status:7}] {code:22} {label:34} {info}")
    time.sleep(0.3)
