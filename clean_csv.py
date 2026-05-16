import csv
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
BACKUP = False  # set to False if you don't want .bak copies

# Try a few common encodings (RU-heavy datasets often use cp1251)
ENCODINGS = ["utf-8-sig", "utf-8", "cp1251", "latin-1"]

def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        # 1) autodetect delimiter with engine="python"
        try:
            return pd.read_csv(
                path,
                engine="python",
                sep=None,                # auto-detect delimiter
                encoding=enc,
                on_bad_lines="warn",     # skip/mark malformed rows
                dtype=str                # keep as text to avoid dtype surprises
            )
        except Exception as e:
            last_err = e
        # 2) common alt delimiter ; if autodetect failed
        try:
            return pd.read_csv(
                path,
                engine="python",
                sep=";",
                encoding=enc,
                on_bad_lines="warn",
                dtype=str
            )
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path} with tried encodings. Last error: {last_err}")

def clean_and_replace_csv(path: Path):
    df = read_csv_robust(path)

    # Normalize column names (optional; comment out if you want to keep exact headers)
    df.columns = [str(c).strip() for c in df.columns]

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    bak_path = path.with_suffix(path.suffix + ".bak")

    # Write back with safe quoting and normalized newlines
    df.to_csv(
        tmp_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,   # quote fields when needed (handles embedded newlines)
        lineterminator="\n"
    )

    # Backup original (optional)
    if BACKUP:
        if bak_path.exists():
            bak_path.unlink()
        path.rename(bak_path)

    # Replace original with cleaned file
    tmp_path.rename(path)

def main():
    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found under ./data")
        return

    ok, fail = 0, 0
    for p in csv_files:
        try:
            clean_and_replace_csv(p)
            ok += 1
            print(f"[OK] {p}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {p} -> {e}", file=sys.stderr)

    print(f"\nDone. Cleaned: {ok}, Failed: {fail}")

if __name__ == "__main__":
    main()
