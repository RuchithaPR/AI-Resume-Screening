"""AI Resume Screening System - Streamlit app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.matcher import SemanticMatcher
from src.parser import extract_text_from_bytes
from src.screener import DEFAULT_WEIGHTS, ScreeningResult, display_name, results_to_dataframe, screen

ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_data"
COMPONENT_LABELS = {"semantic": "Text similarity", "skills": "Skills", "experience": "Experience", "education": "Education"}

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading SBERT model (first run downloads ~90 MB)...")
def load_matcher() -> SemanticMatcher:
    return SemanticMatcher()


def load_sample_jd() -> None:
    st.session_state["jd_text"] = (SAMPLE_DIR / "job_description.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Sidebar: settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    method_label = st.radio(
        "Matching method",
        ["Hybrid: SBERT + skills (recommended)", "Baseline: TF-IDF + skills"],
        help="SBERT understands meaning (e.g. 'built ML models' ≈ 'machine learning experience'). "
             "TF-IDF only compares shared words and is useful as a baseline.",
    )
    method = "sbert" if method_label.startswith("Hybrid") else "tfidf"

    st.subheader("Score weights")
    st.caption("Normalised automatically. Components the job description doesn't specify are ignored.")
    weights = {
        "semantic": st.slider("Text similarity", 0, 100, int(DEFAULT_WEIGHTS["semantic"] * 100)),
        "skills": st.slider("Skills match", 0, 100, int(DEFAULT_WEIGHTS["skills"] * 100)),
        "experience": st.slider("Experience", 0, 100, int(DEFAULT_WEIGHTS["experience"] * 100)),
        "education": st.slider("Education", 0, 100, int(DEFAULT_WEIGHTS["education"] * 100)),
    }

    st.subheader("Display")
    blind = st.checkbox("Blind screening mode", value=False,
                        help="Hides names, emails, phones and file names. Scoring never uses them anyway.")
    top_n = st.number_input("Candidates to show in detail", 1, 50, 5)
    min_score = st.slider("Minimum match % to list", 0, 100, 0)

# --------------------------------------------------------------------------- #
# Main: inputs
# --------------------------------------------------------------------------- #
st.title("📄 AI Resume Screening System")
st.caption("Decision-support tool: ranks candidates against a job description. "
           "It does not make hiring decisions - a recruiter should review every shortlist.")

left, right = st.columns(2)

with left:
    st.subheader("1. Job description")
    st.session_state.setdefault("jd_text", "")
    st.button("Load sample job description", on_click=load_sample_jd)
    st.text_area("Paste the job description", key="jd_text", height=260)
    jd_file = st.file_uploader("...or upload it (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], key="jd_file")

with right:
    st.subheader("2. Resumes")
    resume_files = st.file_uploader("Upload one or more resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    use_samples = st.checkbox("Also include the 5 bundled sample resumes")
    st.caption("Scanned (image-only) PDFs contain no text and will be flagged for manual review.")

jd_text = st.session_state["jd_text"]
if jd_file is not None:
    try:
        jd_text = extract_text_from_bytes(jd_file.getvalue(), jd_file.name)
    except Exception as exc:
        st.error(f"Could not read the job description file: {exc}")

sources: list[tuple[str, bytes]] = [(f.name, f.getvalue()) for f in resume_files]
if use_samples:
    sources += [(p.name, p.read_bytes()) for p in sorted((SAMPLE_DIR / "resumes").glob("*.txt"))]

ready = bool(jd_text.strip()) and bool(sources)
if st.button("🔍 Screen candidates", type="primary", disabled=not ready):
    resumes, errors = [], []
    progress = st.progress(0.0, text="Reading resumes...")
    for i, (name, data) in enumerate(sources):
        try:
            resumes.append((name, extract_text_from_bytes(data, name)))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
        progress.progress((i + 1) / len(sources), text=f"Read {i + 1}/{len(sources)} resumes")
    progress.empty()

    matcher = None
    if method == "sbert":
        try:
            matcher = load_matcher()
        except Exception as exc:
            st.warning(f"SBERT could not be loaded ({exc}). Using TF-IDF instead.")
            method = "tfidf"
    with st.spinner("Scoring candidates..."):
        st.session_state["result"] = screen(jd_text, resumes, weights=weights, method=method, matcher=matcher)
    st.session_state["errors"] = errors
elif not ready:
    st.info("Add a job description and at least one resume to begin.")

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
result: ScreeningResult | None = st.session_state.get("result")
if result:
    for msg in st.session_state.get("errors", []):
        st.error(f"Skipped - {msg}")
    for msg in result.warnings:
        st.warning(msg)

    job = result.job
    st.divider()
    st.header("Results")
    method_name = "SBERT + skills (hybrid)" if result.method == "sbert" else "TF-IDF + skills (baseline)"
    st.caption(f"Method used: **{method_name}** | Effective weights: " +
               ", ".join(f"{COMPONENT_LABELS[k]} {v:.0%}" for k, v in result.weights.items() if v > 0))

    with st.expander(f"What the system understood from the job description: {job.title}", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Required skills**\n\n" + (", ".join(job.required_skills) or "_none detected_"))
        c2.markdown("**Preferred skills**\n\n" + (", ".join(job.preferred_skills) or "_none detected_"))
        c3.markdown(f"**Min. experience:** {f'{job.min_experience:g} years' if job.min_experience else '_not specified_'}\n\n"
                    f"**Education:** {job.education_label}")

    candidates = [c for c in result.candidates if c["score"] >= min_score]
    scores = [c["score"] for c in result.candidates]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resumes screened", len(result.candidates))
    m2.metric("Top match", f"{max(scores):.1f}%" if scores else "-")
    m3.metric("Strong matches (≥75%)", sum(s >= 75 for s in scores))
    m4.metric("Average match", f"{sum(scores) / len(scores):.1f}%" if scores else "-")

    if not candidates:
        st.info("No candidates meet the minimum match % filter.")
    else:
        df = results_to_dataframe(candidates, blind=blind)
        show_cols = [c for c in ["Rank", "Candidate", "Match %", "Verdict", "Semantic %", "Skills %", "Experience %",
                                 "Education %", "Experience (yrs)", "Education", "Matched skills",
                                 "Missing required skills", "Warning"] if c in df.columns]
        st.subheader("Ranked candidates")
        st.dataframe(
            df[show_cols], use_container_width=True, hide_index=True,
            column_config={
                "Match %": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%.1f"),
                "Semantic %": st.column_config.NumberColumn(format="%.0f"),
                "Skills %": st.column_config.NumberColumn(format="%.0f"),
                "Experience %": st.column_config.NumberColumn(format="%.0f"),
                "Education %": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.download_button("⬇️ Download full results (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="resume_screening_results.csv", mime="text/csv")

        # Score breakdown chart
        st.subheader("Why each candidate scored what they did")
        rows = [{"Candidate": f"{c['rank']}. {display_name(c, blind)}", "Component": COMPONENT_LABELS[k], "Points": v}
                for c in candidates[:15] for k, v in c["contributions"].items()]
        fig = px.bar(pd.DataFrame(rows), x="Points", y="Candidate", color="Component", orientation="h",
                     category_orders={"Candidate": [f"{c['rank']}. {display_name(c, blind)}" for c in candidates[:15]]})
        fig.update_layout(yaxis={"autorange": "reversed"}, xaxis_title="Match score (points out of 100)",
                          legend_title_text="", height=60 + 40 * min(len(candidates), 15))
        st.plotly_chart(fig, use_container_width=True)

        # Per-candidate detail
        st.subheader(f"Top {min(int(top_n), len(candidates))} candidates in detail")
        for c in candidates[: int(top_n)]:
            with st.expander(f"#{c['rank']}  {display_name(c, blind)}  -  {c['score']:.1f}%  ({c['verdict']})"):
                if c["warning"]:
                    st.warning(c["warning"])
                st.write(c["explanation"])
                a, b = st.columns(2)
                a.markdown("**✅ Matched required skills**\n\n" + (", ".join(c["matched_required"]) or "_none_"))
                b.markdown("**❌ Missing required skills**\n\n" + (", ".join(c["missing_required"]) or "_none_"))
                if job.preferred_skills:
                    a.markdown("**✅ Matched preferred skills**\n\n" + (", ".join(c["matched_preferred"]) or "_none_"))
                    b.markdown("**➖ Missing preferred skills**\n\n" + (", ".join(c["missing_preferred"]) or "_none_"))
                if c["extra_skills"]:
                    st.caption("Other skills found: " + ", ".join(c["extra_skills"]))
                if not blind:
                    st.caption(f"{c['email']}  {c['phone']}  |  {c['filename']}")
