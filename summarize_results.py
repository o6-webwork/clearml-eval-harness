"""
Summarize scored eval outputs.

Walks a results directory, reads each `*_scored.csv` (and its paired
`*_translations_{dataset}.csv.meta.json`), and emits four files:

  - summary_overall.csv       one row per (model, dataset)
  - summary_by_language.csv   one row per (model, dataset, language)
  - summary_examples.csv      best / worst / polluted exemplars
  - summary.json              all of the above bundled, plus meta

Usage:
  python summarize_results.py --results-dir results_overnight
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re

import pandas as pd

DATASETS = ("mono_sea", "mono_me")
TRUNC = 500
LOW_SAMPLE = 30


def _parse_name(scored_path: str) -> tuple[str, str] | None:
    """`NiuTrans_LMT-60-4B_mono_sea_scored.csv` -> ("NiuTrans/LMT-60-4B", "mono_sea").

    The first underscore separates org from model (slashes aren't allowed in
    filenames, so batch_translate slugifies `org/model` -> `org_model`)."""
    stem = os.path.basename(scored_path).removesuffix("_scored.csv")
    for ds in DATASETS:
        suffix = f"_{ds}"
        if stem.endswith(suffix):
            slug = stem[: -len(suffix)]
            # First underscore is the org/model boundary; later underscores
            # are part of the model name (e.g. LMT-60-4B has none, but
            # HY-MT1.5-1.8B-FP8 has dots, gemma-4-E2B-it has dashes).
            model = slug.replace("_", "/", 1)
            return model, ds
    return None


def _load_meta(results_dir: str, model_slug: str, dataset: str) -> dict:
    path = os.path.join(
        results_dir, f"{model_slug}_translations_{dataset}.csv.meta.json"
    )
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _metric_block(df: pd.DataFrame, threshold: float) -> dict:
    ifeval = df["ifeval"].fillna(0.0)
    comet = df["comet_qe"].fillna(0.0)
    joint = ((ifeval >= 1.0) & (comet >= threshold)).mean()
    return {
        "n_rows": int(len(df)),
        "mean_ifeval": float(ifeval.mean()) if len(df) else float("nan"),
        "mean_comet_qe": float(comet.mean()) if len(df) else float("nan"),
        "median_comet_qe": float(comet.median()) if len(df) else float("nan"),
        "p10_comet_qe": float(comet.quantile(0.10)) if len(df) else float("nan"),
        "p90_comet_qe": float(comet.quantile(0.90)) if len(df) else float("nan"),
        "joint_pass_rate": float(joint) if len(df) else float("nan"),
    }


def _truncate(val, limit: int = TRUNC) -> str:
    if pd.isna(val):
        return ""
    s = str(val)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _examples(df: pd.DataFrame, model: str, dataset: str) -> pd.DataFrame:
    cols = ["language", "source", "hypothesis", "reference", "ifeval", "comet_qe"]
    df = df.copy()
    df["ifeval"] = df["ifeval"].fillna(0.0)
    df["comet_qe"] = df["comet_qe"].fillna(0.0)

    clean = df[df["ifeval"] >= 1.0]
    best = clean.nlargest(5, "comet_qe")[cols].assign(category="best")
    worst = df.nsmallest(5, "comet_qe")[cols].assign(category="worst")
    polluted = df[df["ifeval"] < 1.0].nsmallest(5, "comet_qe")[cols]
    polluted = polluted.assign(category="polluted")

    out = pd.concat([best, worst, polluted], ignore_index=True)
    for c in ("source", "hypothesis", "reference"):
        out[c] = out[c].map(_truncate)
    out.insert(0, "model", model)
    out.insert(1, "dataset", dataset)
    return out[
        ["model", "dataset", "category", "language", "source", "hypothesis",
         "reference", "ifeval", "comet_qe"]
    ]


def summarize(results_dir: str, joint_threshold: float) -> None:
    scored_paths = sorted(glob.glob(os.path.join(results_dir, "*_scored.csv")))
    if not scored_paths:
        raise SystemExit(f"No *_scored.csv under {results_dir}")

    overall_rows: list[dict] = []
    lang_rows: list[dict] = []
    example_frames: list[pd.DataFrame] = []

    for path in scored_paths:
        parsed = _parse_name(path)
        if parsed is None:
            print(f"skip (unparseable name): {path}")
            continue
        model, dataset = parsed
        model_slug = os.path.basename(path).removesuffix(f"_{dataset}_scored.csv")

        df = pd.read_csv(path)
        meta = _load_meta(results_dir, model_slug, dataset)

        n_source = meta.get("n_rows")
        n_errors = meta.get("n_errors")
        error_rate = (
            float(n_errors) / float(n_source)
            if n_source and n_errors is not None
            else float("nan")
        )

        overall_rows.append({
            "model": model,
            "dataset": dataset,
            "n_rows_scored": int(len(df)),
            "n_source_rows": n_source,
            "n_errors": n_errors,
            "error_rate": error_rate,
            **_metric_block(df, joint_threshold),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "req_per_s": meta.get("req_per_s"),
        })
        # n_rows appears twice above (once from _metric_block); dedupe by
        # preferring the _metric_block value — they're equal anyway, but
        # keep it explicit.
        overall_rows[-1].pop("n_rows", None)

        for lang, sub in df.groupby("language", dropna=False):
            lang_rows.append({
                "model": model,
                "dataset": dataset,
                "language": lang,
                **_metric_block(sub, joint_threshold),
                "low_sample": bool(len(sub) < LOW_SAMPLE),
            })

        example_frames.append(_examples(df, model, dataset))

    overall = pd.DataFrame(overall_rows).sort_values(["dataset", "model"])
    by_lang = pd.DataFrame(lang_rows).sort_values(
        ["dataset", "language", "model"]
    )
    examples = pd.concat(example_frames, ignore_index=True)

    overall_path = os.path.join(results_dir, "summary_overall.csv")
    by_lang_path = os.path.join(results_dir, "summary_by_language.csv")
    examples_path = os.path.join(results_dir, "summary_examples.csv")
    json_path = os.path.join(results_dir, "summary.json")

    overall.to_csv(overall_path, index=False)
    by_lang.to_csv(by_lang_path, index=False)
    examples.to_csv(examples_path, index=False)

    bundle = {
        "meta": {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "joint_threshold": joint_threshold,
            "results_dir": os.path.abspath(results_dir),
        },
        "overall": overall.to_dict(orient="records"),
        "by_language": by_lang.to_dict(orient="records"),
        "examples": examples.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(bundle, f, indent=2, default=str)

    print(f"wrote {overall_path}")
    print(f"wrote {by_lang_path}")
    print(f"wrote {examples_path}")
    print(f"wrote {json_path}")
    print()
    display_cols = [
        "model", "dataset", "n_rows_scored", "mean_ifeval", "mean_comet_qe",
        "joint_pass_rate", "error_rate", "req_per_s",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(overall[display_cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results_overnight")
    ap.add_argument("--joint-threshold", type=float, default=0.80,
                    help="COMET-QE threshold for joint_pass_rate (default 0.80)")
    args = ap.parse_args()
    summarize(args.results_dir, args.joint_threshold)


if __name__ == "__main__":
    main()
