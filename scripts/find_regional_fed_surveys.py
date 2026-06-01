"""Locate the full set of regional Fed manufacturing surveys on FRED."""
import sys
import yaml
from fredapi import Fred

QUERIES = [
    "Empire State Manufacturing",
    "Philadelphia Fed Manufacturing",
    "Richmond Fed Manufacturing",
    "Dallas Fed Manufacturing",
    "Kansas City Fed Manufacturing",
    "Atlanta Fed Business",
    "Minneapolis Fed",
]


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    fred = Fred(api_key=cfg["api_key"])

    for q in QUERIES:
        print(f"\n=== {q} ===")
        try:
            hits = fred.search(q, limit=15)
        except Exception as e:
            print(f"  search failed: {e}")
            continue
        if hits is None or hits.empty:
            print("  no hits")
            continue
        cols = [c for c in ["id", "title", "frequency_short", "observation_start", "observation_end", "last_updated"] if c in hits.columns]
        for _, row in hits[cols].iterrows():
            obs_end = str(row.get("observation_end", ""))[:10]
            last_upd = str(row.get("last_updated", ""))[:10]
            print(f"  {row['id']:30s}  {row.get('frequency_short','')}  end={obs_end}  upd={last_upd}  | {row['title'][:70]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
