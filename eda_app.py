"""
Translation Eval EDA — Streamlit + DuckDB

A small visual-inspection tool over scored translation CSVs. Handles the
~500k-row scale of a full multi-model run without pulling everything into
pandas (DuckDB reads the CSVs in place).

Install once:
  .venv/bin/pip install streamlit duckdb plotly

Run:
  .venv/bin/streamlit run eda_app.py -- --results-dir results_overnight

Three tabs:
  1. Summary   — per-model × per-dataset headline table + distributions
  2. Browse    — filter on model / dataset / language / score ranges,
                 click a row to see its full source + hypothesis + reference
  3. Compare   — pick a source (id2) and see every model's hypothesis
                 side-by-side
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


SOURCE_JSONL_FILES = {
    "bi_ar_id": "bi_ar_id.jsonl",
    "bi_en_ar": "bi_en_ar.jsonl",
    "bi_en_id": "bi_en_id.jsonl",
    "mono_me":  "mono_me.jsonl",
    "mono_sea": "mono_sea.jsonl",
    "multi":    "multi.jsonl",
}

SLANG_LEX_PATH = os.path.join(os.path.dirname(__file__), "data", "kamus_alay.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    # streamlit forwards args after `--` to the script.
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_overnight")
    # streamlit injects its own args when --server.* flags are used; ignore unknown.
    return ap.parse_known_args(sys.argv[1:])[0]


ARGS = _parse_args()
DEFAULT_DIR = ARGS.results_dir


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def _duck() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


@st.cache_data(show_spinner="Loading scored CSVs via DuckDB...")
def load_scored(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "*_scored.csv")))
    if not paths:
        return pd.DataFrame()
    # union_by_name handles minor column-order differences across models.
    # `filename=true` adds a `filename` column we parse into dataset + model_slug.
    sql = f"""
        SELECT *
        FROM read_csv_auto(
            '{os.path.join(results_dir, "*_scored.csv")}',
            filename = true,
            union_by_name = true
        )
    """
    df = _duck().execute(sql).df()

    def _dataset(fn: str) -> str:
        b = os.path.basename(fn)
        if b.endswith("_mono_sea_scored.csv"):
            return "mono_sea"
        if b.endswith("_mono_me_scored.csv"):
            return "mono_me"
        return "unknown"

    df["dataset"] = df["filename"].map(_dataset)
    # 'model' column is present in the source CSV; keep that as the canonical
    # name. Drop the full path from view.
    df["_source_file"] = df["filename"].map(os.path.basename)
    df = df.drop(columns=["filename"])
    return df


@st.cache_data(show_spinner="Loading source JSONL files...")
def load_source_jsonl(source_dir: str) -> pd.DataFrame:
    frames = []
    for ds, fname in SOURCE_JSONL_FILES.items():
        fp = os.path.join(source_dir, fname)
        if not os.path.isfile(fp):
            continue
        part = pd.read_json(fp, lines=True)
        part["dataset"] = ds
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["text_chars"] = out["text"].fillna("").str.len()
    out["text_words"] = out["text"].fillna("").str.split().str.len()
    out["translation_chars"] = out["translation"].fillna("").str.len()
    return out


@st.cache_data(show_spinner="Computing Indonesian slang density (mono_sea)...")
def compute_slang_density_mono_sea(source_dir: str) -> pd.DataFrame:
    fp = os.path.join(source_dir, SOURCE_JSONL_FILES["mono_sea"])
    if not os.path.isfile(fp) or not os.path.isfile(SLANG_LEX_PATH):
        return pd.DataFrame()
    from slang_density import density, load_lexicon
    lex = load_lexicon(SLANG_LEX_PATH)
    rows = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d, h, n = density(r.get("text", ""), lex)
            rows.append({"density": d, "hits": h, "n_tokens": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preview(text, n: int = 80) -> str:
    if pd.isna(text):
        return ""
    s = str(text).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _metric_block(df: pd.DataFrame, threshold: float = 0.80) -> dict:
    if df.empty:
        return {"n": 0}
    ifeval = df["ifeval"].fillna(0.0)
    comet = df["comet_qe"].fillna(0.0)
    return {
        "n": len(df),
        "mean_ifeval": float(ifeval.mean()),
        "mean_comet_qe": float(comet.mean()),
        "median_comet_qe": float(comet.median()),
        "p10_comet_qe": float(comet.quantile(0.10)),
        "p90_comet_qe": float(comet.quantile(0.90)),
        "joint_pass": float(((ifeval >= 1.0) & (comet >= threshold)).mean()),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Translation Eval EDA", layout="wide")
st.title("Translation Eval EDA")

with st.sidebar:
    st.caption("Data")
    results_dir = st.text_input("Results dir", value=DEFAULT_DIR)
    if not os.path.isdir(results_dir):
        st.error(f"Not a directory: {results_dir}")
        st.stop()

df = load_scored(results_dir)
if df.empty:
    st.error(f"No *_scored.csv files under {results_dir}")
    st.stop()

with st.sidebar:
    st.caption(f"Loaded {len(df):,} rows from {df['_source_file'].nunique()} file(s)")
    st.divider()
    st.caption("Filters (apply to Browse + Compare)")
    all_models = sorted(df["model"].dropna().unique().tolist())
    all_datasets = sorted(df["dataset"].unique().tolist())
    all_langs = sorted(df["language"].dropna().unique().tolist())

    sel_datasets = st.multiselect("Dataset", all_datasets, default=all_datasets)
    sel_models = st.multiselect("Model", all_models, default=all_models)
    sel_langs = st.multiselect("Language", all_langs, default=[])
    ifeval_filter = st.selectbox("IFEval", ["any", "clean (1)", "polluted (0)"], index=0)
    comet_range = st.slider("COMET-QE range", 0.0, 1.0, (0.0, 1.0), 0.01)
    has_slang = "slang_density" in df.columns
    if has_slang:
        slang_range = st.slider(
            "Slang density range", 0.0, 1.0, (0.0, 1.0), 0.01,
            help="Fraction of source tokens in Kamus Alay. Indonesian-specific; non-id rows sit near 0.",
        )
        slang_min_hits = st.number_input(
            "Min slang hits", value=0, min_value=0,
            help="Absolute slang-token count — density on very short rows is noisy.",
        )
    else:
        slang_range = (0.0, 1.0)
        slang_min_hits = 0
    text_q = st.text_input("Text search (source / hypothesis)", value="")
    joint_threshold = st.number_input(
        "Joint pass threshold (COMET-QE)", value=0.80, min_value=0.0, max_value=1.0, step=0.05,
    )


def apply_filters(base: pd.DataFrame) -> pd.DataFrame:
    m = pd.Series(True, index=base.index)
    if sel_datasets:
        m &= base["dataset"].isin(sel_datasets)
    if sel_models:
        m &= base["model"].isin(sel_models)
    if sel_langs:
        m &= base["language"].isin(sel_langs)
    if ifeval_filter == "clean (1)":
        m &= base["ifeval"].fillna(0.0) >= 1.0
    elif ifeval_filter == "polluted (0)":
        m &= base["ifeval"].fillna(0.0) < 1.0
    lo, hi = comet_range
    m &= base["comet_qe"].fillna(0.0).between(lo, hi)
    if "slang_density" in base.columns:
        sd_lo, sd_hi = slang_range
        m &= base["slang_density"].fillna(0.0).between(sd_lo, sd_hi)
        if slang_min_hits > 0:
            m &= base["slang_hits"].fillna(0) >= slang_min_hits
    if text_q:
        q = text_q.lower()
        m &= (
            base["source"].fillna("").str.lower().str.contains(q, regex=False)
            | base["hypothesis"].fillna("").str.lower().str.contains(q, regex=False)
        )
    return base[m]


filt = apply_filters(df)

tab_summary, tab_browse, tab_compare, tab_source = st.tabs(
    ["Summary", "Browse", "Compare", "Source EDA"]
)


# ---------------------------------------------------------------------------
# Summary tab
# ---------------------------------------------------------------------------

with tab_summary:
    st.caption(f"Computing over {len(filt):,} filtered rows "
               f"(of {len(df):,} total).")

    group_rows = []
    for (model, dataset), sub in filt.groupby(["model", "dataset"]):
        group_rows.append({
            "model": model, "dataset": dataset,
            **_metric_block(sub, threshold=joint_threshold),
        })
    summary = pd.DataFrame(group_rows).sort_values(["dataset", "model"])

    st.subheader("Headline table")
    if summary.empty:
        st.info("No rows match the current filters.")
    else:
        st.dataframe(
            summary.style.format({
                c: "{:.4f}" for c in summary.columns
                if c not in ("model", "dataset", "n")
            }).format({"n": "{:,}"}),
            use_container_width=True, hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("COMET-QE distribution")
            fig = px.box(
                filt, x="model", y="comet_qe", color="dataset",
                points=False, height=420,
            )
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Mean IFEval (cleanliness)")
            fig = px.bar(
                summary, x="model", y="mean_ifeval", color="dataset",
                barmode="group", height=420, range_y=(0, 1),
            )
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-language COMET-QE (mean)")
        per_lang = (filt.groupby(["model", "dataset", "language"])["comet_qe"]
                        .mean().reset_index())
        fig = px.bar(
            per_lang.sort_values("language"),
            x="language", y="comet_qe", color="model",
            facet_row="dataset", barmode="group", height=520, range_y=(0, 1),
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Browse tab
# ---------------------------------------------------------------------------

with tab_browse:
    st.caption(f"{len(filt):,} rows after filters.")

    col_sort, col_limit = st.columns([2, 1])
    with col_sort:
        sort_by = st.selectbox(
            "Sort by",
            ["comet_qe ↑ (worst first)", "comet_qe ↓ (best first)",
             "ifeval ↑ (polluted first)", "source length ↓"],
        )
    with col_limit:
        limit = st.number_input("Show top N", value=500, min_value=50,
                                max_value=20000, step=50)

    view = filt.copy()
    if sort_by.startswith("comet_qe ↑"):
        view = view.sort_values("comet_qe", ascending=True, na_position="first")
    elif sort_by.startswith("comet_qe ↓"):
        view = view.sort_values("comet_qe", ascending=False, na_position="last")
    elif sort_by.startswith("ifeval ↑"):
        view = view.sort_values(["ifeval", "comet_qe"], ascending=[True, True])
    else:
        view = view.assign(_len=view["source"].fillna("").str.len())\
                   .sort_values("_len", ascending=False).drop(columns="_len")

    view = view.head(int(limit))
    display_cols = ["model", "dataset", "language", "source_preview",
                    "hyp_preview", "gt_preview", "ifeval", "comet_qe"]
    if "slang_density" in view.columns:
        display_cols += ["slang_density", "slang_hits"]
    display_cols += ["id1", "id2"]
    display = view.assign(
        source_preview=view["source"].map(lambda s: _preview(s, 100)),
        hyp_preview=view["hypothesis"].map(lambda s: _preview(s, 100)),
        gt_preview=view["reference"].map(lambda s: _preview(s, 100)),
    )[display_cols].rename(columns={"gt_preview": "google_translate"})

    event = st.dataframe(
        display, use_container_width=True, hide_index=True, height=420,
        on_select="rerun", selection_mode="single-row",
    )

    sel = event.selection.rows if hasattr(event, "selection") else []
    if sel:
        row = view.iloc[sel[0]]
        st.divider()
        st.subheader("Row detail")
        meta_cols = st.columns(5)
        meta_cols[0].metric("Model", row["model"])
        meta_cols[1].metric("Dataset", row["dataset"])
        meta_cols[2].metric("Language", str(row["language"]))
        meta_cols[3].metric("IFEval", f"{row['ifeval']:.0f}" if pd.notna(row["ifeval"]) else "—")
        meta_cols[4].metric("COMET-QE", f"{row['comet_qe']:.3f}" if pd.notna(row["comet_qe"]) else "—")

        st.markdown("**Source**")
        st.code(row["source"] or "", language=None)
        st.markdown("**Hypothesis** (model)")
        st.code(row["hypothesis"] or "", language=None)
        st.markdown("**Reference** (Google Translate baseline)")
        st.code(row["reference"] or "", language=None)
        extra = f"id1={row['id1']}  ·  id2={row['id2']}"
        if "slang_density" in row.index and pd.notna(row["slang_density"]):
            extra += f"  ·  slang_density={row['slang_density']:.2f}  hits={int(row['slang_hits'])}/{int(row['n_tokens'])}"
        st.caption(extra)
    else:
        st.caption("Click a row above to see full source / model hypothesis / Google Translate reference.")


# ---------------------------------------------------------------------------
# Compare tab — same source across every model
# ---------------------------------------------------------------------------

with tab_compare:
    st.caption("Pick a source and see every model's hypothesis side-by-side.")

    # Identify candidate sources — rows where ≥2 models translated the same id.
    # Compute from full df (not filt) so the chooser isn't empty when filters
    # exclude some models.
    keys = ["id1", "id2", "language", "dataset"]
    count_by_key = (
        df.groupby(keys, dropna=False)["model"].nunique().reset_index(name="n_models")
    )
    candidates = count_by_key[count_by_key["n_models"] >= 2]

    if candidates.empty:
        st.info("No rows with the same id covered by ≥2 models.")
    else:
        # Quick filters to narrow the candidate list
        c1, c2, c3 = st.columns(3)
        cand_datasets = sorted(candidates["dataset"].unique().tolist())
        cand_langs = sorted(candidates["language"].dropna().unique().tolist())
        with c1:
            d_sel = st.selectbox("Dataset", ["(any)"] + cand_datasets, key="cmp_ds")
        with c2:
            l_sel = st.selectbox("Language", ["(any)"] + cand_langs, key="cmp_lg")
        with c3:
            only_disagreements = st.checkbox(
                "Only IFEval disagreements", value=False,
                help="Rows where at least one model said clean and another polluted",
            )

        cand = candidates.copy()
        if d_sel != "(any)":
            cand = cand[cand["dataset"] == d_sel]
        if l_sel != "(any)":
            cand = cand[cand["language"] == l_sel]

        if only_disagreements:
            # Join to df for ifeval variance per key
            ifev = (df.groupby(keys)["ifeval"].agg(["min", "max"]).reset_index())
            disagree = ifev[(ifev["min"] < 1.0) & (ifev["max"] >= 1.0)][keys]
            cand = cand.merge(disagree, on=keys)

        st.caption(f"{len(cand):,} candidate source(s).")
        if cand.empty:
            st.stop()

        # Render the selector as a compact text so users can scan
        cand = cand.head(500).reset_index(drop=True)  # cap for UI perf
        # Attach a preview of the source for the selector label
        preview = (df.drop_duplicates(keys)[keys + ["source"]]
                     .merge(cand, on=keys))
        preview["label"] = preview.apply(
            lambda r: f"[{r['dataset']}/{r['language']}]  {_preview(r['source'], 100)}",
            axis=1,
        )
        choice = st.selectbox("Source", preview["label"], key="cmp_choice")
        sel_row = preview[preview["label"] == choice].iloc[0]

        st.markdown("**Source**")
        st.code(sel_row["source"] or "", language=None)

        # Grab all models' rows for this key
        hit = df.merge(
            pd.DataFrame([sel_row[keys].to_dict()]), on=keys, how="inner"
        )
        hit = hit[["model", "hypothesis", "reference", "ifeval", "comet_qe"]]\
                  .sort_values("comet_qe", ascending=False)

        ref = hit["reference"].dropna().iloc[0] if not hit["reference"].dropna().empty else ""
        if ref:
            st.markdown("**Reference** (Google Translate baseline)")
            st.code(ref, language=None)

        st.markdown("**Model outputs** (sorted by COMET-QE)")
        for _, r in hit.iterrows():
            with st.container(border=True):
                top = st.columns([3, 1, 1])
                top[0].markdown(f"**{r['model']}**")
                top[1].metric("IFEval",
                              f"{r['ifeval']:.0f}" if pd.notna(r["ifeval"]) else "—")
                top[2].metric("COMET-QE",
                              f"{r['comet_qe']:.3f}" if pd.notna(r["comet_qe"]) else "—")
                st.code(r["hypothesis"] or "", language=None)


# ---------------------------------------------------------------------------
# Source EDA tab — stats on the raw jsonl inputs
# ---------------------------------------------------------------------------

with tab_source:
    st.caption("EDA on the raw JSONL source files (pre-translation).")

    source_dir = st.text_input(
        "Source dir (jsonl files)", value="data", key="src_dir",
        help="Directory containing mono_*.jsonl / bi_*.jsonl / multi.jsonl",
    )
    if not os.path.isdir(source_dir):
        st.error(f"Not a directory: {source_dir}")
        st.stop()

    src = load_source_jsonl(source_dir)
    if src.empty:
        st.info(f"No known jsonl files under {source_dir}")
        st.stop()

    st.caption(
        f"Loaded {len(src):,} rows across {src['dataset'].nunique()} dataset(s)."
    )

    # --- Headline table --------------------------------------------------
    st.subheader("Headline table")
    head = (
        src.groupby("dataset")
           .agg(
               n_rows=("text", "size"),
               mean_chars=("text_chars", "mean"),
               median_chars=("text_chars", "median"),
               p90_chars=("text_chars", lambda s: s.quantile(0.90)),
               mean_words=("text_words", "mean"),
               pct_code_switched=("is_code_switched", "mean"),
               mean_lang_count=("lang_count", "mean"),
               mean_sem_sim=("semantic_similarity", "mean"),
           )
           .reset_index()
           .sort_values("dataset")
    )
    st.dataframe(
        head.style.format({
            "n_rows": "{:,}",
            "mean_chars": "{:.1f}",
            "median_chars": "{:.0f}",
            "p90_chars": "{:.0f}",
            "mean_words": "{:.1f}",
            "pct_code_switched": "{:.1%}",
            "mean_lang_count": "{:.2f}",
            "mean_sem_sim": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # --- Language distribution ------------------------------------------
    st.subheader("Language distribution")
    st.caption(
        "Counts each language occurrence in `lang_codes` — a code-switched row "
        "with `ar,id` contributes one to `ar` and one to `id`."
    )
    lang_df = (
        src.assign(lang=src["lang_codes"].fillna("").str.split(","))
           .explode("lang")
    )
    lang_df["lang"] = lang_df["lang"].str.strip()
    lang_df = lang_df[lang_df["lang"] != ""]
    lang_counts = (lang_df.groupby(["dataset", "lang"]).size()
                           .reset_index(name="n"))
    fig = px.bar(
        lang_counts.sort_values(["dataset", "n"], ascending=[True, False]),
        x="lang", y="n", color="dataset", barmode="group", height=420,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- Length distribution --------------------------------------------
    st.subheader("Source text length")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            src, x="text_chars", color="dataset", nbins=80,
            log_y=True, height=380, barmode="overlay", opacity=0.55,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="chars")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(
            src, x="dataset", y="text_chars", points=False, height=380,
            log_y=True,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="chars (log)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Code-switching / lang_count ------------------------------------
    st.subheader("Code-switching & language count")
    c1, c2 = st.columns(2)
    with c1:
        cs = (src.groupby("dataset")["is_code_switched"].mean()
                 .reset_index(name="pct_code_switched"))
        fig = px.bar(
            cs, x="dataset", y="pct_code_switched",
            range_y=(0, 1), height=360,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        lc = (src.groupby(["dataset", "lang_count"]).size()
                 .reset_index(name="n"))
        fig = px.bar(
            lc, x="lang_count", y="n", color="dataset",
            barmode="group", height=360,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- Semantic similarity (source vs translation baseline) -----------
    st.subheader("Semantic similarity (source ↔ translation baseline)")
    fig = px.box(
        src, x="dataset", y="semantic_similarity", points=False, height=360,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- Slang density for mono_sea (Indonesian) ------------------------
    st.subheader("Slang density — mono_sea (Indonesian)")
    st.caption(
        "Fraction of source word tokens in Kamus Alay. Non-Indonesian rows "
        "sit near 0; density on short rows is noisy (see `hits`)."
    )
    slang = compute_slang_density_mono_sea(source_dir)
    if slang.empty:
        st.info("mono_sea.jsonl or data/kamus_alay.json not found.")
    else:
        min_tokens = st.number_input(
            "Min n_tokens (filter short rows)", value=5, min_value=0,
            key="src_slang_min_tokens",
        )
        filt_slang = slang[slang["n_tokens"] >= min_tokens]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("n rows", f"{len(filt_slang):,}")
            st.metric("mean density",
                      f"{filt_slang['density'].mean():.3f}"
                      if not filt_slang.empty else "—")
            st.metric("median density",
                      f"{filt_slang['density'].median():.3f}"
                      if not filt_slang.empty else "—")
            st.metric("p90 density",
                      f"{filt_slang['density'].quantile(0.90):.3f}"
                      if not filt_slang.empty else "—")
            st.metric("% rows with ≥1 hit",
                      f"{(filt_slang['hits'] > 0).mean() * 100:.1f}%"
                      if not filt_slang.empty else "—")
        with c2:
            if not filt_slang.empty:
                fig = px.histogram(
                    filt_slang, x="density", nbins=50, height=360,
                )
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
