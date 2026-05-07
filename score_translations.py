"""
Score Translations
==================
Reads every *_translations*.csv under OUTPUT_DIR and appends quality scores,
writing *_scored.csv next to each input.

Metric selection (SCORE_METRICS env var, comma-separated subset):
  - ifeval — zero-shot NLI, cross-encoder/nli-deberta-v3-large
  - qe     — xCOMET-QE (reference-free, Unbabel/XCOMET-XL without the ref input)
  - ref    — xCOMET (reference-based, src + mt + ref)
Default: SCORE_METRICS=ifeval,qe

Uncomment the optional metrics below to additionally run LaBSE, BERTScore,
or CometKiwi.

Runs inside the scorer container (see docker-compose-scorer.yml) which has
torch + transformers + unbabel-comet pre-installed.

Environment variables:
  OUTPUT_DIR        folder of translation CSVs. Default ./results
  SCORE_METRICS     comma-separated subset of {ifeval, qe, ref}.
                    Default: ifeval,qe
  COMET_BATCH_SIZE  default 16 (FP16, A5000 16GB). Drop to 8 if OOM.
  COMET_FP16        1 (default) | 0
  HF_TOKEN          required for gated models like Unbabel/XCOMET-XL
"""

import glob
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Lazy-loaded scoring models (loaded once per process, reused across CSVs)
# ---------------------------------------------------------------------------

_nli_pipeline = None
_xcomet_model = None


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        import torch
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-large",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _nli_pipeline


def _get_xcomet_model():
    """Load XCOMET-XL once per process. The checkpoint is ~3.5 GB and takes
    15–45 s to deserialise, so reloading per CSV was the bulk of scoring time.

    FP16 by default — halves VRAM (the 3.5B-param XLM-R-XL encoder is the
    bottleneck) which lets us roughly double the batch size versus FP32.
    Set COMET_FP16=0 to fall back to full precision if scores look off."""
    global _xcomet_model
    if _xcomet_model is None:
        import torch
        from comet import download_model, load_from_checkpoint
        # Enable Tensor Core matmul on Ampere+ (A5000, A100, RTX 30/40 etc.).
        # Trades ~2 ulps of precision for a noticeable speedup on the XLM-R-XL
        # encoder; "high" = TF32 paths, "medium" = full bf16 reductions.
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision(
                os.environ.get("COMET_MATMUL_PRECISION", "high")
            )
        print("  Loading XCOMET-XL (first call — may take 30s)...")
        model_path = download_model("Unbabel/XCOMET-XL")
        _xcomet_model = load_from_checkpoint(model_path)
        if os.environ.get("COMET_FP16", "1") == "1" and torch.cuda.is_available():
            _xcomet_model = _xcomet_model.half()
            print("  XCOMET-XL cast to FP16.")
    return _xcomet_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

IFEVAL_HYPOTHESIS_TEMPLATE = (
    "This translation output is {} — it contains only the translation with "
    "no preamble, postamble, or explanation."
)


def score_ifeval(translations: list[str]) -> list[float]:
    """Zero-shot NLI classifier flags polluted outputs (preamble, explanation).
    Returns 1.0 (clean) / 0.0 (polluted) per example. Good ≥ 0.80.

    Empty hypotheses short-circuit to 0.0 and are not sent to the pipeline."""
    nli = _get_nli_pipeline()
    batch_size = int(os.environ.get("IFEVAL_BATCH_SIZE", "32"))

    texts = [(i, h.strip()) for i, h in enumerate(translations)]
    non_empty = [(i, t) for i, t in texts if t]

    scores = [0.0] * len(translations)
    if not non_empty:
        return scores

    results = nli(
        [t for _, t in non_empty],
        candidate_labels=["clean", "polluted"],
        hypothesis_template=IFEVAL_HYPOTHESIS_TEMPLATE,
        batch_size=batch_size,
    )
    # pipeline returns a single dict for a single input, a list otherwise.
    if isinstance(results, dict):
        results = [results]
    for (idx, _), res in zip(non_empty, results):
        scores[idx] = 1.0 if res["labels"][0] == "clean" else 0.0
    return scores


def _xcomet_predict(data: list[dict]) -> list[float]:
    # FP16 roughly halves activation memory, so 16 is a safe default on a
    # 16 GB A5000. Drop back to 8 (or lower) if you see OOMs; 32–48 fits on
    # 24+ GB cards. num_workers keeps the GPU fed during tokenisation;
    # 0 (default) bottlenecks throughput on long inputs.
    batch_size = int(os.environ.get("COMET_BATCH_SIZE", "16"))
    num_workers = int(os.environ.get("COMET_NUM_WORKERS", "2"))
    model = _get_xcomet_model()
    return model.predict(
        data,
        batch_size=batch_size,
        gpus=1,
        num_workers=num_workers,
    ).scores


def score_xcomet(
    sources: list[str], translations: list[str], references: list[str]
) -> list[float]:
    """Reference-based xCOMET. Uses src + mt + ref."""
    data = [
        {"src": s, "mt": t, "ref": r}
        for s, t, r in zip(sources, translations, references)
    ]
    return _xcomet_predict(data)


def score_xcomet_qe(
    sources: list[str], translations: list[str]
) -> list[float]:
    """Reference-free xCOMET. Uses src + mt only.

    Useful when references are synthetic (e.g. machine-translated) and
    shouldn't be trusted as ground truth. XCOMET-XL is a unified model — it
    handles both modes; the ref key is simply omitted from the input dict."""
    data = [{"src": s, "mt": t} for s, t in zip(sources, translations)]
    return _xcomet_predict(data)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)

INPUT_GLOB = os.environ.get("INPUT_GLOB") or os.path.join(
    RESULTS_DIR, "*_translations*.csv"
)

SRC_COL = "source"
HYP_COL = "hypothesis"
REF_COL = "reference"

VALID_METRICS = {"ifeval", "qe", "ref"}


def _resolve_score_metrics() -> set[str]:
    """Pick which metrics to compute for this scorer invocation.

    Reads SCORE_METRICS env var (comma-separated subset of VALID_METRICS).
    Default: {ifeval, qe}.
    """
    raw = os.environ.get("SCORE_METRICS", "").strip()
    if raw:
        metrics = {m.strip().lower() for m in raw.split(",") if m.strip()}
        unknown = metrics - VALID_METRICS
        if unknown:
            raise SystemExit(
                f"Unknown metrics in SCORE_METRICS={raw!r}: {sorted(unknown)}. "
                f"Valid: {sorted(VALID_METRICS)}"
            )
        if not metrics:
            raise SystemExit(
                f"SCORE_METRICS={raw!r} resolved to an empty set."
            )
        return metrics

    return {"ifeval", "qe"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_file(input_csv: str, metrics: set[str] | None = None) -> str:
    print(f"\n{'=' * 60}")
    print(f"SCORING: {input_csv}")
    print("=" * 60)

    if metrics is None:
        metrics = _resolve_score_metrics()
    print(f"  Metrics: {sorted(metrics)}")

    df = pd.read_csv(input_csv)

    n_before = len(df)
    df = df.dropna(subset=[HYP_COL]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing hypothesis")

    sources = df[SRC_COL].fillna("").tolist()
    hypotheses = df[HYP_COL].fillna("").tolist()
    references = df[REF_COL].fillna("").tolist()

    if "ifeval" in metrics:
        print(f"  Running IFEval on {len(hypotheses)} rows...")
        df["ifeval"] = score_ifeval(hypotheses)

    # qe   → comet_qe (src + mt — reference-free; default because
    #                  synthetic / MT references aren't ground truth)
    # ref  → comet    (src + mt + ref)
    if "ref" in metrics:
        print(f"  Running xCOMET (ref-based) on {len(hypotheses)} rows...")
        df["comet"] = score_xcomet(sources, hypotheses, references)
    if "qe" in metrics:
        print(f"  Running xCOMET (reference-free) on {len(hypotheses)} rows...")
        df["comet_qe"] = score_xcomet_qe(sources, hypotheses)

    # Uncomment to add optional metrics (you'll need the corresponding
    # pip packages already in the scorer image — sentence-transformers and
    # bert-score are pre-installed):
    # from sentence_transformers import SentenceTransformer, util
    # labse = SentenceTransformer("sentence-transformers/LaBSE")
    # src_emb = labse.encode(sources, convert_to_tensor=True)
    # hyp_emb = labse.encode(hypotheses, convert_to_tensor=True)
    # df["labse"] = util.cos_sim(src_emb, hyp_emb).diagonal().tolist()
    #
    # from bert_score import score as bertscore
    # _, _, f1 = bertscore(hypotheses, references, lang="en", verbose=False)
    # df["bertscore_f1"] = f1.tolist()

    stem = os.path.splitext(os.path.basename(input_csv))[0].replace("_translations", "")
    out_path = os.path.join(RESULTS_DIR, f"{stem}_scored.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)

    n = len(df)
    print(f"\n  Scored → {out_path}")
    print(f"  Averages (n={n}):")
    if "ifeval" in df.columns:
        print(f"    IFEval         : {df['ifeval'].mean():.4f}  (good ≥ 0.80)")
    if "comet" in df.columns:
        print(f"    xCOMET (ref)   : {df['comet'].mean():.4f}")
    if "comet_qe" in df.columns:
        print(f"    xCOMET-QE (qe) : {df['comet_qe'].mean():.4f}")
    return out_path


if __name__ == "__main__":
    inputs = sorted(glob.glob(INPUT_GLOB))
    if not inputs:
        raise SystemExit(f"No files matched {INPUT_GLOB}")
    metrics = _resolve_score_metrics()
    print(f"Found {len(inputs)} translation CSV(s) to score")
    print(f"Metrics: {sorted(metrics)}")
    for path in inputs:
        score_file(path, metrics=metrics)
