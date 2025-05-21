#!/usr/bin/env python
"""
01_clean.py
───────────
Usage:
    python flashcamp/scripts/01_clean.py --in data/raw.csv --out data/gold/v1.parquet

• Drops duplicate startup_id rows (keeps latest)
• Ensures all 99 columns exist (adds missing as NaN/0)
• Casts numeric columns, lower-cases traffic-light strings
"""

import argparse
import pandas as pd
from pathlib import Path
import sys, json

ROOT = Path(__file__).resolve().parents[1]          # flashcamp/
# Read metrics from the correct frontend path
METRICS_PATH = ROOT / "frontend/src/constants/metrics.json"
if not METRICS_PATH.is_file():
    # Fallback to older path if src/constants doesn't exist yet
    METRICS_PATH = ROOT / "frontend/constants/metrics.json"
    if not METRICS_PATH.is_file():
      print(f"Error: Cannot find metrics.json at {ROOT / 'frontend/src/constants/metrics.json'} or {ROOT / 'frontend/constants/metrics.json'}")
      sys.exit(1)

METRICS = json.loads(METRICS_PATH.read_text())

NUM_COLS  = {m["name"] for m in METRICS if m["type"] == "number"}
LIST_COLS = {m["name"] for m in METRICS if m["type"] == "list"}
CHECKBOX_COLS = {m["name"] for m in METRICS if m["type"] == "checkbox"}

def main(in_path: Path, out_path: Path) -> None:
    df = pd.read_csv(in_path)

    # 1. add missing columns
    all_metric_names = {m["name"] for m in METRICS}
    for col_name in all_metric_names:
        if col_name not in df.columns:
            df[col_name] = [] if col_name in LIST_COLS else pd.NA

    # Ensure the DataFrame only contains columns defined in METRICS
    df = df[list(all_metric_names)]

    # 2. deduplicate on startup_id (keep last)
    if "startup_id" in df.columns:
        df = df.sort_values("startup_id").drop_duplicates("startup_id", keep="last")

    # 3. numeric casting
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. boolean casting for checkbox columns
    true_values = ['true', 'yes', '1', True, 1]
    for c in CHECKBOX_COLS:
        # Convert to lowercase string, handle NaNs, then check against true_values
        df[c] = df[c].astype(str).str.lower().isin(true_values)

    # 5. traffic-light normalisation
    df.replace({"Green":"green", "Amber":"amber", "Red":"red",
                "Light-green":"light-green", "Deep-red":"deep-red"}, inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ cleaned → {out_path}  ({len(df)} rows)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="outp", required=True)
    args = p.parse_args()
    # Adjust path relative to workspace root
    main(Path(args.inp), Path(args.outp)) 