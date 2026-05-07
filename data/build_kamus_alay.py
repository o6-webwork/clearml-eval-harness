"""One-shot: reduce the upstream Kamus Alay CSV to a slang->formal dict.

Run once after fetching `colloquial-indonesian-lexicon.csv` from
nasalsabila/kamus-alay. Writes `kamus_alay.json` alongside, which
`prepare_slang_split.py` loads at runtime.

The upstream CSV has ~15k rows and duplicates (one slang token can map
to multiple formal candidates across rows). We keep the first mapping
seen — the sieve only needs the *set of slang tokens*, so collisions
don't matter.
"""

from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "colloquial-indonesian-lexicon.csv")
DST = os.path.join(HERE, "kamus_alay.json")


def main() -> None:
    mapping: dict[str, str] = {}
    with open(SRC, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slang = (row.get("slang") or "").strip().lower()
            formal = (row.get("formal") or "").strip().lower()
            if not slang or slang == formal:
                continue
            mapping.setdefault(slang, formal)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=0, sort_keys=True)
    print(f"wrote {DST}  ({len(mapping):,} entries)")


if __name__ == "__main__":
    main()
