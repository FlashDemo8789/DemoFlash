import argparse, pandas as pd, pandera as pa
from flashcamp.backend.schema.models import StartupInput
from pandera.typing import DataFrame
from pathlib import Path

class StartupDFSchema(pa.DataFrameModel):
    # auto-generate cols from Pydantic using pandas-schema
    # fallback simple check: just ensure all required cols are present
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    missing = set(StartupInput.schema()["properties"]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df.to_parquet(args.out, index=False)
    print("👍 Data validated and written to", args.out)

if __name__ == "__main__":
    main() 