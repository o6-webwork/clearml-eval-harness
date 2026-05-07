"""Indonesian slang density helpers.

A row's slang density is the fraction of its word tokens that appear in
the Kamus Alay lexicon. `hits` and `n_tokens` are also returned so
callers can filter on absolute counts (density on a 3-word row is noisy).

Lexical-only: catches abbreviations / elongations / lexicalised slang,
misses syntactic informality. See ROADMAP.
"""

from __future__ import annotations

import json
import re

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN.findall(text)]


def load_lexicon(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return set(json.load(f).keys())


def density(text: str, lex: set[str]) -> tuple[float, int, int]:
    toks = tokens(text)
    if not toks:
        return 0.0, 0, 0
    hits = sum(1 for t in toks if t in lex)
    return hits / len(toks), hits, len(toks)
