"""Split the Indonesian subset of an eval JSONL into slang-heavy and
formal tracks.

The premise (see ROADMAP "Dual-track slang benchmark"): XCOMET-QE
penalises correct colloquial Indonesian translations because the slang
tokens don't align with its embeddings. To test that hypothesis we need
two tracks evaluated differently — reference-free QE on formal rows,
reference-based COMET on slang rows with pseudo-gold references.

This script is the sieve for step 1: for every Indonesian row in the
input JSONL, count how many of its whitespace-separated tokens appear in
the Kamus Alay slang lexicon. Top N rows by density go into the slang
split; the rest into the formal split. Non-Indonesian rows are ignored.

Lexical limitation (noted in ROADMAP): the lexicon catches
abbreviations / elongations / lexicalised slang. It misses purely
*syntactic* informality (particle drop, code-switching into English
with standard tokens). That's acceptable for v1.

Usage:
  python prepare_slang_split.py \
      --input data/mono_sea.jsonl \
      --lexicon data/kamus_alay.json \
      --out-dir data \
      --n-slang 1000
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from slang_density import density as _row_density, load_lexicon, tokens as _tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/mono_sea.jsonl",
                    help="JSONL with id Indonesian rows (default data/mono_sea.jsonl)")
    ap.add_argument("--lexicon", default="data/kamus_alay.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--lang-code", default="id",
                    help="lang_codes value to filter on (default 'id')")
    ap.add_argument("--n-slang", type=int, default=1000,
                    help="rows to put in the slang split (top-density)")
    ap.add_argument("--min-tokens", type=int, default=5,
                    help="skip rows with fewer tokens than this (density is noisy on tiny inputs)")
    ap.add_argument("--slang-name", default=None,
                    help="output stem for slang split; defaults to <input_stem>_<lang>_slang")
    ap.add_argument("--formal-name", default=None,
                    help="output stem for formal split; defaults to <input_stem>_<lang>_formal")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon)
    print(f"lexicon: {len(lex):,} slang tokens")

    stem = os.path.basename(args.input).removesuffix(".jsonl")
    slang_name = args.slang_name or f"{stem}_{args.lang_code}_slang"
    formal_name = args.formal_name or f"{stem}_{args.lang_code}_formal"

    os.makedirs(args.out_dir, exist_ok=True)
    slang_path = os.path.join(args.out_dir, f"{slang_name}.jsonl")
    formal_path = os.path.join(args.out_dir, f"{formal_name}.jsonl")

    rows: list[tuple[float, int, int, dict]] = []  # (density, hits, n_tokens, row)
    seen = Counter()
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seen[row.get("lang_codes")] += 1
            if row.get("lang_codes") != args.lang_code:
                continue
            d, h, n = _row_density(row.get("text", ""), lex)
            if n < args.min_tokens:
                continue
            rows.append((d, h, n, row))

    print(f"input lang breakdown: {dict(seen)}")
    print(f"eligible {args.lang_code} rows (>= {args.min_tokens} tokens): {len(rows):,}")

    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    n_slang = min(args.n_slang, len(rows))

    with open(slang_path, "w", encoding="utf-8") as f:
        for d, h, n, row in rows[:n_slang]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(formal_path, "w", encoding="utf-8") as f:
        for d, h, n, row in rows[n_slang:]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {slang_path}  ({n_slang:,} rows)")
    print(f"wrote {formal_path}  ({len(rows) - n_slang:,} rows)")

    if n_slang:
        top = rows[:n_slang]
        densities = [r[0] for r in top]
        hits = [r[1] for r in top]
        print(f"slang split density: min={min(densities):.3f} median={densities[len(densities)//2]:.3f} max={max(densities):.3f}")
        print(f"slang split hits/row: min={min(hits)} median={sorted(hits)[len(hits)//2]} max={max(hits)}")
        print(f"cutoff density (row {n_slang}): {rows[n_slang - 1][0]:.3f}")
        print("\nfirst 3 slang rows (preview):")
        for _, _, _, row in top[:3]:
            t = (row.get("text") or "").replace("\n", " ")
            print(f"  - {t[:140]}{'…' if len(t) > 140 else ''}")


if __name__ == "__main__":
    main()
