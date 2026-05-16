import pandas as pd
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union


ENCODINGS = ["utf-8-sig", "utf-8", "cp1251", "latin-1"]


def read_csv(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        # 1) autodetect delimiter
        try:
            return pd.read_csv(path, engine="python", sep=None, encoding=enc, dtype=str, on_bad_lines="skip")
        except Exception as e:
            last_err = e
        # 2) fallback to semicolon
        try:
            return pd.read_csv(path, engine="python", sep=";", encoding=enc, dtype=str, on_bad_lines="skip")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path} with tried encodings. Last error: {last_err}")


def read_excel(path: Path) -> List[Tuple[str, pd.DataFrame]]:
    xl = pd.ExcelFile(path)
    sheets = []
    for name in xl.sheet_names:
        df = xl.parse(name, dtype=str)
        sheets.append((name, df))
    return sheets


def is_empty_cell(v: Optional[str]) -> bool:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def row_all_empty(row: Iterable[Optional[str]]) -> bool:
    return all(is_empty_cell(x) for x in row)


def trim_empty_borders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Drop fully-empty rows
    mask_rows = ~df.apply(lambda r: row_all_empty(r), axis=1)
    df = df.loc[mask_rows]
    # Drop fully-empty columns
    mask_cols = ~df.apply(lambda c: row_all_empty(c), axis=0)
    df = df.loc[:, mask_cols]
    return df


def split_into_tables(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split by blank-row separators into multiple sub-frames."""
    if df.empty:
        return []

    # Ensure string dtype for uniformity
    df = df.astype("object")
    blocks = []
    start = None

    for i in range(len(df)):
        row = df.iloc[i, :]
        if row_all_empty(row):
            if start is not None:
                sub = df.iloc[start:i, :]
                sub = trim_empty_borders(sub)
                if not sub.empty:
                    blocks.append(sub)
                start = None
        else:
            if start is None:
                start = i

    if start is not None:
        sub = df.iloc[start:len(df), :]
        sub = trim_empty_borders(sub)
        if not sub.empty:
            blocks.append(sub)

    return blocks


def looks_like_header(cells: List[str], next_row: Optional[List[str]]) -> bool:
    """Heuristic: not-mostly-numeric, unique-ish, non-empty."""
    vals = [("" if x is None else str(x).strip()) for x in cells]
    if not vals or all(v == "" for v in vals):
        return False

    non_empty = [v for v in vals if v != ""]
    if len(non_empty) < max(1, len(vals) // 2):  # too sparse
        return False
    
    _numeric_re = re.compile(r"^\s*[-+]?\d+([.,]\d+)?\s*$")

    # not mostly numeric
    numeric_count = sum(1 for v in non_empty if _numeric_re.match(v))
    if numeric_count > len(non_empty) * 0.5:
        return False

    # uniqueness
    uniq_ratio = len(set(non_empty)) / len(non_empty)
    if uniq_ratio < 0.7:
        return False

    # compare with next row types (header often differs from next row which is numeric-ish)
    if next_row is not None:
        next_vals = [("" if x is None else str(x).strip()) for x in next_row]
        # if header is much less numeric than next row, good sign
        next_non_empty = [v for v in next_vals if v != ""]
        next_numeric = sum(1 for v in next_non_empty if _numeric_re.match(v))
        if next_non_empty:
            if numeric_count <= (0.5 * next_numeric):
                return True

    return True


def detect_header_and_columns(df: pd.DataFrame) -> Tuple[Optional[List[str]], List[List[str]]]:
    """Return (header or None, rows) with unified column width."""
    if df.empty:
        return None, []

    # Convert to list-of-lists (strings)
    raw = df.fillna("").astype(str).values.tolist()

    # Determine max width across rows
    width = max(len(r) for r in raw) if raw else 0
    norm_rows = [ (r + [""] * (width - len(r))) for r in raw ]

    header = None
    if norm_rows:
        first = norm_rows[0]
        nxt = norm_rows[1] if len(norm_rows) > 1 else None
        if looks_like_header(first, nxt):
            header = [c.strip() for c in first]
            data_rows = norm_rows[1:]
        else:
            data_rows = norm_rows
    else:
        data_rows = []

    return header, data_rows


def esc(s: str) -> str:
    """Escape pipes and normalize whitespace/newlines for markdown cells."""
    if s is None:
        return ""
    s = str(s).replace("|", r"\|")
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")  # flatten newlines
    return s.strip()

def md_table(header: Optional[List[str]], rows: List[List[str]]) -> str:
    # unify width from header/rows
    width = 0
    if header:
        width = max(width, len(header))
    for r in rows:
        width = max(width, len(r))
    if width == 0:
        return ""

    lines = []
    if header:
        h = [esc(x) for x in (header + [""] * (width - len(header)))]
        lines.append("| " + " | ".join(h) + " |")
        lines.append("| " + " | ".join(["---"] * width) + " |")
    else:
        # no header: still add separator line to make a valid table?
        # Common practice is to synthesize a blank header row:
        lines.append("| " + " | ".join([""] * width) + " |")
        lines.append("| " + " | ".join(["---"] * width) + " |")

    for r in rows:
        rr = [esc(x) for x in (r + [""] * (width - len(r)))]
        lines.append("| " + " | ".join(rr) + " |")
    return "\n".join(lines)


def convert_df_to_markdown_tables(df: pd.DataFrame) -> List[str]:
    tables = split_into_tables(df)
    outputs = []
    for t in tables:
        header, rows = detect_header_and_columns(t)
        outputs.append(md_table(header, rows))
    return outputs

def process_file_test(path: Path) -> List[Tuple[str, str]]:
    ext = path.suffix.lower()
    if ext == ".csv":
        sheets = [("csv", read_csv(path))]
    elif ext in (".xlsx", ".xls"):
        sheets = read_excel(path)
    else:
        return []

    out: List[Tuple[str, str]] = []
    output_str = ""
    base = path.stem
    for sheet, df in sheets:
        mds = convert_df_to_markdown_tables(df)
        if not mds:
            continue
        output_str += f"## Sheet {sheet}\n\n"
        for i, md in enumerate(mds, start=1):
            if not md.strip():
                continue
            table_id = f"{sheet}#table{i}"
            out.append((table_id, md))
            output_str += f"### Table {i}\n{md}\n\n"
    if output_str.strip():
        fname = f"{base}.txt"
        dest = path.with_name(fname)  # same directory, .txt extension
        dest.write_text(output_str, encoding="utf-8")
    return out


def process_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        sheets = [("csv", read_csv(path))]
    elif ext in (".xlsx", ".xls"):
        sheets = read_excel(path)
    else:
        return ""

    output_lines = []
    for sheet, df in sheets:
        mds = convert_df_to_markdown_tables(df)
        if not mds:
            continue
        output_lines.append(f"## Sheet: {sheet}")
        for i, md in enumerate(mds, start=1):
            output_lines.append(f"### Table {i}")
            output_lines.append(md)
            output_lines.append("")  # Empty line between tables

    return "\n".join(output_lines)
