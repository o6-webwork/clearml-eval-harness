"""Add slang_density / slang_hits / n_tokens columns to scored CSVs.

Rather than pre-splitting the JSONL into slang/formal files, this tool
annotates every row of every *_scored.csv (or any CSV with a `source`
column) with its slang-density score. The EDA app can then filter on
the column directly.

The lexicon is Indonesian-specific — non-Indonesian rows will naturally
get near-zero density, but the columns are populated for every row
regardless (keeps the filter composable).

Usage:
  python enrich_slang_density.py --results-dir results_overnight
  python enrich_slang_density.py --glob 'results_overnight/*_scored.csv' --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from slang_density import density, load_lexicon

NEW_COLS = ("slang_density", "slang_hits", "n_tokens")


def enrich(path: str, lex: set[str], text_col: str, overwrite: bool) -> tuple[int, int]:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise SystemExit(f"{path}: no column '{text_col}' (have: {list(df.columns)})")
    if all(c in df.columns for c in NEW_COLS) and not overwrite:
        return len(df), 0
    rows = df[text_col].fillna("").map(lambda s: density(s, lex))
    df["slang_density"] = [r[0] for r in rows]
    df["slang_hits"] = [r[1] for r in rows]
    df["n_tokens"] = [r[2] for r in rows]
    df.to_csv(path, index=False)
    return len(df), 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=None,
                    help="enrich every *_scored.csv under this dir")
    ap.add_argument("--glob", default=None,
                    help="explicit glob; overrides --results-dir")
    ap.add_argument("--lexicon", default="data/kamus_alay.json")
    ap.add_argument("--text-col", default="source",
                    help="column to compute density from (default 'source')")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if columns already exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.glob:
        paths = sorted(glob.glob(args.glob))
    elif args.results_dir:
        paths = sorted(glob.glob(os.path.join(args.results_dir, "*_scored.csv")))
    else:
        raise SystemExit("pass --results-dir or --glob")
    if not paths:
        raise SystemExit("no CSVs matched")

    lex = load_lexicon(args.lexicon)
    print(f"lexicon: {len(lex):,} slang tokens")
    print(f"files: {len(paths)}")

    total_rows = 0
    total_written = 0
    for p in paths:
        if args.dry_run:
            print(f"  would enrich {p}")
            continue
        n, changed = enrich(p, lex, args.text_col, args.overwrite)
        total_rows += n
        total_written += changed
        marker = "enriched" if changed else "skip (already has columns)"
        print(f"  {marker}: {p}  ({n:,} rows)")

    if not args.dry_run:
        print(f"\ntotal rows: {total_rows:,}  files written: {total_written}")


if __name__ == "__main__":
    main()
