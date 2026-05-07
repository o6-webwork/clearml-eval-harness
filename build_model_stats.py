"""
Build Model Stats
=================
Regenerates the tables under `## Coverage matrix`, `## Quality + format per split`,
and `## Speed` in `submissions/sprint-1.5/team-{1,2}/model-stats.md` from the
scored CSVs + meta JSONs produced by `run_all_evals.py`.

Idempotent: the script replaces everything between `<!-- stats:begin -->` and
`<!-- stats:end -->` markers; anything outside those markers is left alone.
The markers are injected on first run if they don't exist yet.

Splits are inferred from the scored CSV filename. Expected pattern:
    {model_safe}_{split}_scored.csv
e.g. `google_gemma-4-E2B-it_mono_me_scored.csv` -> split = "mono_me".

Missing columns (src-copy flag, slang bucket) render as "—" rather than
zeros — absence is explicit so a reviewer can see which detectors ran.

Usage:
    python bench/orchestration/build_model_stats.py \
        --results bench/orchestration/results_full \
        --stats-files \
            submissions/sprint-1.5/team-1/model-stats.md \
            submissions/sprint-1.5/team-2/model-stats.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd


BEGIN = "<!-- stats:begin -->"
END = "<!-- stats:end -->"

KNOWN_SPLITS = ("mono_me", "mono_sea", "bilingual")
SCORED_SUFFIX = "_scored.csv"
TRANSLATIONS_PREFIX = "_translations_"


def parse_scored_filename(path: str) -> tuple[str, str] | None:
    """`foo_bar_mono_me_scored.csv` -> ("foo_bar", "mono_me"). Returns None if
    no known split suffix matches."""
    stem = os.path.basename(path)[: -len(SCORED_SUFFIX)]
    for split in KNOWN_SPLITS:
        suffix = f"_{split}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    return None


def find_meta(results_dir: str, model_safe: str, split: str) -> dict | None:
    path = os.path.join(
        results_dir, f"{model_safe}{TRANSLATIONS_PREFIX}{split}.csv.meta.json"
    )
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def pct(series: pd.Series, value) -> str:
    if series is None or series.empty:
        return "—"
    return f"{100.0 * (series == value).mean():.1f}"


def mean_or_dash(series: pd.Series | None) -> str:
    if series is None or series.empty:
        return "—"
    return f"{series.mean():.4f}"


def load_scored(results_dir: str) -> list[dict]:
    """One row per (model, split) with aggregated stats."""
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"*{SCORED_SUFFIX}"))):
        parsed = parse_scored_filename(path)
        if parsed is None:
            continue
        model_safe, split = parsed
        df = pd.read_csv(path)
        src_copy_col = df["source_copy"] if "source_copy" in df.columns else None
        slang_col = df["slang_bucket"] if "slang_bucket" in df.columns else None

        formal_comet_qe = slang_comet_qe = None
        if slang_col is not None and "comet_qe" in df.columns:
            formal_comet_qe = df.loc[slang_col == "formal", "comet_qe"]
            slang_comet_qe = df.loc[slang_col == "slang", "comet_qe"]

        rows.append({
            "model_safe": model_safe,
            "split": split,
            "n": len(df),
            "comet_qe": mean_or_dash(df["comet_qe"]) if "comet_qe" in df.columns else "—",
            "ifeval": mean_or_dash(df["ifeval"]) if "ifeval" in df.columns else "—",
            "src_copy_pct": pct(src_copy_col, True) if src_copy_col is not None else "—",
            "slang_pct": pct(slang_col, "slang") if slang_col is not None else "—",
            "formal_comet_qe": mean_or_dash(formal_comet_qe),
            "slang_comet_qe": mean_or_dash(slang_comet_qe),
            "meta": find_meta(results_dir, model_safe, split),
        })
    return rows


def render_coverage(rows: list[dict]) -> str:
    models = sorted({r["model_safe"] for r in rows})
    by_key = {(r["model_safe"], r["split"]): r for r in rows}
    lines = ["| Model | mono_me | mono_sea | bilingual |",
             "|-------|---------|----------|-----------|"]
    for m in models:
        cells = []
        for split in ("mono_me", "mono_sea", "bilingual"):
            r = by_key.get((m, split))
            cells.append(f"ran ({r['n']})" if r else "missing")
        lines.append(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines) if models else "_No scored CSVs found._"


def render_quality_split(rows: list[dict], split: str) -> str:
    filtered = [r for r in rows if r["split"] == split]
    if not filtered:
        return f"_No scored CSVs for `{split}`._"
    header = (
        "| Model | n | xCOMET-QE | IFEval | src-copy % | slang % "
        "| formal xCOMET-QE | slang xCOMET-QE |\n"
        "|-------|---|-----------|--------|------------|---------"
        "|------------------|-----------------|"
    )
    body = "\n".join(
        f"| {r['model_safe']} | {r['n']} | {r['comet_qe']} | {r['ifeval']} "
        f"| {r['src_copy_pct']} | {r['slang_pct']} "
        f"| {r['formal_comet_qe']} | {r['slang_comet_qe']} |"
        for r in sorted(filtered, key=lambda r: r["model_safe"])
    )
    return f"{header}\n{body}"


def render_speed(rows: list[dict]) -> str:
    speed_rows = [r for r in rows if r["meta"]]
    if not speed_rows:
        return "_No meta JSON found next to any scored CSV._"
    header = (
        "| Model | Split | elapsed_s | req/s | concurrency | n_errors |\n"
        "|-------|-------|-----------|-------|-------------|----------|"
    )
    def fmt(meta, key, fmtstr):
        v = meta.get(key)
        return fmtstr.format(v) if isinstance(v, (int, float)) else "—"
    body = "\n".join(
        f"| {r['model_safe']} | {r['split']} "
        f"| {fmt(r['meta'], 'elapsed_seconds', '{:.1f}')} "
        f"| {fmt(r['meta'], 'req_per_s', '{:.2f}')} "
        f"| {r['meta'].get('concurrency', '—')} "
        f"| {r['meta'].get('n_errors', '—')} |"
        for r in sorted(speed_rows, key=lambda r: (r["model_safe"], r["split"]))
    )
    return f"{header}\n{body}"


def scorer_commit() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "score_translations.py"],
            cwd=here,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or "—"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "—"


def render_block(results_dir: str, rows: list[dict]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = [
        BEGIN,
        f"- **Last generated:** {ts}",
        f"- **Results directory:** `{results_dir}`",
        f"- **Scorer commit:** `{scorer_commit()}`",
        "",
        "## Coverage matrix",
        "",
        render_coverage(rows),
        "",
        "## Quality + format per split",
        "",
        "### mono_me",
        "",
        render_quality_split(rows, "mono_me"),
        "",
        "### mono_sea",
        "",
        render_quality_split(rows, "mono_sea"),
        "",
        "### bilingual",
        "",
        render_quality_split(rows, "bilingual"),
        "",
        "## Speed",
        "",
        render_speed(rows),
        END,
    ]
    return "\n".join(parts)


def inject(stats_path: str, block: str) -> None:
    with open(stats_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    if BEGIN in text and END in text:
        new_text = re.sub(
            re.escape(BEGIN) + r".*?" + re.escape(END),
            block,
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"

    if new_text != text:
        with open(stats_path, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        print(f"  Updated: {stats_path}")
    else:
        print(f"  No change: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_full"
        ),
        help="Scorer output directory.",
    )
    parser.add_argument(
        "--stats-files",
        nargs="+",
        required=True,
        help="model-stats.md files to update (one per team).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results):
        print(f"Results dir not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    rows = load_scored(args.results)
    print(f"Loaded {len(rows)} (model, split) row(s) from {args.results}")

    block = render_block(args.results, rows)
    for stats_path in args.stats_files:
        inject(stats_path, block)


if __name__ == "__main__":
    main()
