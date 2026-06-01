"""Probe ALFRED vintage coverage for the series planned for the soft-vs-hard paper.

For each series, report:
  - whether vintage dates exist (i.e. series is in ALFRED, not just FRED)
  - first vintage date, last vintage date, total vintage count
  - first observation date, last observation date

Run: python scripts/check_vintage_coverage.py [path/to/config.yaml]
The config file only needs the `api_key` field.
"""
import sys
import yaml
from fredapi import Fred

SERIES = {
    "hard": {
        "GDPC1":     "Real GDP (quarterly)",
        "INDPRO":    "Industrial Production",
        "UNRATE":    "Unemployment Rate",
        "CPIAUCSL":  "CPI All Items",
        "PCEPILFE":  "Core PCE Price Index",
        "PAYEMS":    "Nonfarm Payrolls",
        "FEDFUNDS":  "Federal Funds Rate",
        "GS10":      "10-Year Treasury Yield",
        "GS3M":      "3-Month Treasury Yield",
        "TB3MS":     "3-Month T-Bill (alt for spread)",
    },
    "soft": {
        "UMCSENT":     "Michigan Consumer Sentiment",
        "CSCICP03USM665S": "OECD Consumer Confidence (US)",
        "NAPM":        "ISM Manufacturing PMI (legacy)",
        "NAPMNMI":     "ISM Non-Manufacturing PMI (legacy)",
        "USEPUINDXM":  "BBD Economic Policy Uncertainty (monthly)",
        "USEPUINDXD":  "BBD Economic Policy Uncertainty (daily)",
    },
    "regime": {
        "VIXCLS":  "CBOE VIX (daily)",
        "NFCI":    "Chicago Fed NFCI",
    },
    "ism_substitutes": {
        "GACDISA066MSFRBNY":  "Empire State Mfg General Activity (NY Fed)",
        "GACDFSA066MSFRBPHI": "Philly Fed Mfg General Activity",
        "MFGCSA066MSFRBRIC":  "Richmond Fed Mfg Composite",
        "BACTSAMFRBDAL":      "Dallas Fed Mfg Business Activity",
        "MFGCMIKCFRB":        "Kansas City Fed Mfg Composite",
        "USSLIND":            "Leading Index for the US (Philly Fed)",
        "DRTSCILM":           "Senior Loan Officer Survey: tightening C&I (large/medium)",
    },
}


def probe(fred, code):
    try:
        s = fred.get_series(code)
    except Exception as e:
        return {"status": "missing", "error": str(e).splitlines()[0]}
    try:
        vdates = fred.get_series_vintage_dates(code)
    except Exception as e:
        return {
            "status": "no_vintages",
            "obs_first": str(s.index.min().date()),
            "obs_last":  str(s.index.max().date()),
            "obs_n":     int(s.notna().sum()),
            "error":     str(e).splitlines()[0],
        }
    return {
        "status":     "ok",
        "obs_first":  str(s.index.min().date()),
        "obs_last":   str(s.index.max().date()),
        "obs_n":      int(s.notna().sum()),
        "vint_first": str(min(vdates).date()) if len(vdates) else None,
        "vint_last":  str(max(vdates).date()) if len(vdates) else None,
        "vint_n":     len(vdates),
    }


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    fred = Fred(api_key=config["api_key"])

    for block, group in SERIES.items():
        print(f"\n=== {block.upper()} ===")
        for code, label in group.items():
            r = probe(fred, code)
            if r["status"] == "missing":
                print(f"  [MISSING]    {code:20s} {label}  -> {r['error']}")
            elif r["status"] == "no_vintages":
                print(f"  [NO VINTAGE] {code:20s} {label}  obs {r['obs_first']}..{r['obs_last']} (n={r['obs_n']})")
            else:
                print(
                    f"  [OK]         {code:20s} {label}  "
                    f"obs {r['obs_first']}..{r['obs_last']} (n={r['obs_n']})  "
                    f"vint {r['vint_first']}..{r['vint_last']} (n={r['vint_n']})"
                )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(path)
