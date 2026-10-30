"""Streamlit front end: single-resume analysis, multi-resume ranking, evaluation results, methodology."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from resume_intel.assessment import DISCLAIMER
from resume_intel.extraction import ocr_available
from resume_intel.jobs import parse_job_description
from resume_intel.pipeline import analyze_batch, analyze_file, rank
from resume_intel.reports import candidate_html, candidate_markdown, ranking_csv, ranking_html, ranking_json
from resume_intel.semantic import encoder_status, get_encoder

ROOT = Path(__file__).parent
st.set_page_config(page_title="ResumeIQ", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading semantic model…")
def encoder():
    return get_encoder()


@st.cache_data(show_spinner="Analyzing…", max_entries=16)
def cached_single(data: bytes, filename: str, jd_text: str, _enc):
    return analyze_file(data, filename, parse_job_description(jd_text), _enc)


def sample_jd():
    p = ROOT / "samples" / "sample_job_description.txt"
    return p.read_text() if p.exists() else ""


st.title("📄 ResumeIQ")
st.caption("Structured resume parsing · hybrid job matching · explainable ATS-style assessment · candidate ranking · improvement recommendations")

with st.sidebar:
    st.header("Job description")
    if "jd" not in st.session_state:
        st.session_state.jd = ""
    if st.button("Load sample job description"):
        st.session_state.jd = sample_jd()
    jd_text = st.text_area("Paste the job description", key="jd", height=300)
    anonymize = st.toggle("Blind review (hide names)", value=False, help="Replaces candidate names with 'Candidate N' in tables and reports. Names never affect scores.")
    enc = encoder()
    st.markdown(f"**Semantic encoder:** `{enc.name}`")
    if enc.name.startswith("concept"):
        st.warning("MiniLM model not available locally; using a weaker fallback. See README.")
    st.markdown(f"**OCR for scanned PDFs:** {'available' if ocr_available() else 'not installed (text PDFs/DOCX only)'}")
    st.info("🔒 Everything runs locally. Resume text is processed in memory, is never written to disk and is not sent to any external service. Analyses stay in this server process's memory until you clear them or stop the app.")
    if st.button("Clear uploaded data from memory"):
        st.cache_data.clear()
        for k in ("rank_sig", "rank_res"):
            st.session_state.pop(k, None)
        st.rerun()

jd = parse_job_description(jd_text) if jd_text.strip() else None
tab_one, tab_rank, tab_eval, tab_about = st.tabs(["Analyze one resume", "Rank candidates", "Evaluation", "Method & limits"])


def show_job_summary():
    with st.expander("How the job description was understood", expanded=False):
        c1, c2 = st.columns(2)
        c1.write("**Required:** " + (", ".join(jd.required_skills) or "—"))
        c2.write("**Preferred:** " + (", ".join(jd.preferred_skills) or "—"))
        st.write(f"Minimum experience: {jd.min_years or 'not stated'} yrs · Degree level required: {jd.education_level or 'not stated'} · Certifications mentioned: {len(jd.certifications)}")
        for w in jd.warnings:
            st.warning(w)


def show_analysis(a, key):
    if not a.ok:
        st.error(f"{a.filename}: {a.error}")
        return
    m, r, s = a.match, a.resume, a.assessment
    for w in a.extraction_warnings:
        st.warning(w)
    c = st.columns(5)
    c[0].metric("Compatibility", f"{m.score:.0f}/100", m.band)
    c[1].metric("Required skills evidenced", f"{100 * m.required_coverage:.0f}%")
    c[2].metric("Completeness", f"{s.completeness_score:.0f}/100")
    c[3].metric("Readability", f"{s.readability_score:.0f}/100")
    c[4].metric("Original baseline", f"{m.baseline['score']:.0f}")
    st.caption(DISCLAIMER)
    if m.confidence != "high":
        st.warning(f"Parse confidence is **{m.confidence}**. Items reported as missing may be extraction gaps.")
    t = st.tabs(["Score breakdown", "Skills", "Evidence", "Assessment", "Recommendations", "Parsed resume", "Download"])
    with t[0]:
        df = pd.DataFrame([{"Factor": d.label, "Score %": None if d.score is None else round(100 * d.score), "Weight %": round(100 * d.weight), "Points": d.contribution, "Why": d.detail} for d in m.dimensions])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("Factor")["Points"])
        st.caption("Weights of non-applicable factors are redistributed. Points sum to the compatibility score.")
    with t[1]:
        rows = [{"Skill": k.skill, "Level": k.level, "Result": k.status + (" (list only)" if k.listed_only else ""), "Via": k.via or "", "Possibly not extracted": k.uncertain, "Note": k.note} for k in m.skill_matches]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        extra = sorted(r.skill_names - {k.skill for k in m.skill_matches})
        st.write("**Other skills on the resume:** " + (", ".join(extra) or "—"))
    with t[2]:
        st.subheader("Relevant experience")
        for e in m.experience_evidence or []:
            st.markdown(f"**{e['title'] or 'Role'}** {('· ' + e['organization']) if e['organization'] else ''} {e['dates']} — relevance {e['relevance']:.2f}  \nMatched: {', '.join(e['matched_skills']) or '—'}")
            for ln in e["lines"]:
                st.markdown(f"> {ln}")
        st.subheader("Relevant projects")
        for e in m.project_evidence or []:
            st.markdown(f"**{e['title']}** — relevance {e['relevance']:.2f}  \nMatched: {', '.join(e['matched_skills']) or '—'}")
            for ln in e["lines"]:
                st.markdown(f"> {ln}")
        st.subheader("Qualifications")
        st.write("Education: " + m.education["detail"])
        for cr in m.certifications:
            st.write(f"Certification ({cr['level']}): {cr['requirement'][:100]} → **{cr['status']}**" + (f" ({cr['resume_certification']})" if cr["resume_certification"] else ""))
        st.subheader("Closest resume evidence for each job requirement")
        st.dataframe(pd.DataFrame(m.semantic_evidence), hide_index=True, use_container_width=True)
    with t[3]:
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("#### ✅ Strengths")
            for x in s.strengths:
                st.markdown(f"- {x['text']}" + (f"  \n  _“{x['evidence'][:160]}”_" if x.get("evidence") else ""))
        with a2:
            st.markdown("#### ⚠️ Gaps (not evidenced)")
            for x in s.gaps:
                st.markdown(f"- {x['text']}")
        with a3:
            st.markdown("#### ❓ Uncertain (may be extraction limits)")
            for x in s.uncertainties:
                st.markdown(f"- {x['text']}")
        st.markdown("#### ATS-style checks")
        icon = {"pass": "✅", "warn": "🟡", "fail": "❌"}
        st.dataframe(pd.DataFrame([{"": icon[c["status"]], "Check": c["label"], "Detail": c["detail"]} for c in s.checks]), hide_index=True, use_container_width=True)
        st.caption("Generic ATS-style practices only; this tool makes no claim about any specific commercial ATS.")
    with t[4]:
        st.caption("improve_existing = reword what is already there · add_if_true = add genuine information you may have omitted · develop = a real gap · restructure = make parsing easier. Never add anything that is not true.")
        for kind, label in (("improve_existing", "Improve existing content"), ("add_if_true", "Add genuine omitted information (only if true)"), ("develop", "Skills / qualifications to develop"), ("restructure", "Document structure")):
            items = [x for x in s.recommendations if x["kind"] == kind]
            if items:
                st.markdown(f"#### {label}")
                for x in items:
                    with st.expander(f"[{x['priority']}] {x['title']}"):
                        st.write(x["detail"])
                        if x.get("evidence"):
                            st.markdown(f"_From your resume:_ {x['evidence']}")
                        if x.get("example"):
                            st.code(x["example"], language=None)
    with t[5]:
        d = r.to_dict()
        st.write(f"**Name:** {'(hidden)' if anonymize else (r.name or '—')} ({r.name_confidence} confidence) · **E-mail:** {'(hidden)' if anonymize else (r.email or '—')} · **Phone:** {'(hidden)' if anonymize else (r.phone or '—')}")
        st.write("**Links:** " + ("(hidden)" if anonymize else (", ".join(f"{k}: {v}" for k, v in r.links.items()) if r.links else "—")))
        st.write(f"**Estimated experience:** {(r.total_experience_months or 0) / 12:.1f} yrs" if r.total_experience_months is not None else "**Estimated experience:** not determinable")
        st.write("**Sections found:** " + ", ".join(r.sections_found) + " · **Missing:** " + (", ".join(r.sections_missing) or "none"))
        for w in r.warnings:
            st.warning(w)
        for title, key in (("Experience", "experience"), ("Education", "education"), ("Projects", "projects"), ("Certifications", "certifications")):
            st.markdown(f"#### {title}")
            st.dataframe(pd.json_normalize(d[key]).drop(columns=["raw"], errors="ignore").astype(str) if d[key] else pd.DataFrame(), use_container_width=True)
        st.write("**Skills (listed ✓ = in a skills section):** " + ", ".join(f"{x.name}{'✓' if x.listed else ''}" for x in r.skills))
        if r.other_listed_skills:
            st.write("**Other listed items (outside the skill taxonomy):** " + ", ".join(r.other_listed_skills))
        if not anonymize:
            with st.expander("Extracted raw text"):
                st.text(r.text)
    with t[6]:
        st.download_button("HTML report (printable)", candidate_html(a, jd.title, anonymize), f"assessment_{key}.html", "text/html", key=f"h{key}")
        st.download_button("Markdown report", candidate_markdown(a, jd.title, anonymize), f"assessment_{key}.md", "text/markdown", key=f"m{key}")
        st.download_button("JSON (full analysis)", json.dumps(a.to_dict(anonymize_as="Candidate" if anonymize else None), indent=2, default=str), f"analysis_{key}.json", "application/json", key=f"j{key}")


with tab_one:
    f = st.file_uploader("Upload a resume (PDF or DOCX)", type=["pdf", "docx"], key="one")
    if not jd:
        st.info("Paste a job description in the sidebar to begin.")
    elif f:
        show_job_summary()
        a = cached_single(f.getvalue(), f.name, jd_text, enc)
        show_analysis(a, "single")

with tab_rank:
    files = st.file_uploader("Upload multiple resumes (PDF / DOCX)", type=["pdf", "docx"], accept_multiple_files=True, key="many")
    if not jd:
        st.info("Paste a job description in the sidebar to begin.")
    elif files:
        sig = (tuple((f.name, f.size) for f in files), jd_text)
        if st.session_state.get("rank_sig") != sig:
            bar = st.progress(0.0, text="Analyzing…")
            res = analyze_batch([(f.name, f.getvalue()) for f in files], jd_text, enc, progress=lambda i, n, nm: bar.progress(i / max(n, 1), text=f"Analyzing {nm} ({i}/{n})"))
            bar.empty()
            st.session_state.update(rank_sig=sig, rank_res=res)
        res = st.session_state.rank_res
        rows = rank(res, anonymize=anonymize)
        ok = [r for r in rows if r["status"] == "ok"]
        errs = [r for r in rows if r["status"] != "ok"]
        if errs:
            st.warning(f"{len(ok)} analyzed · {len(errs)} failed")
        else:
            st.success(f"{len(ok)} analyzed")
        for r in errs:
            st.error(f"{r['file']}: {r['error']}")  # file names are shown to the uploader; they are replaced when blind review is on
        if ok:
            df = pd.DataFrame(ok)
            f1, f2, f3, f4 = st.columns(4)
            min_score = f1.slider("Minimum score", 0, 100, 0)
            min_cov = f2.slider("Minimum required-skill coverage %", 0, 100, 0)
            need = f3.multiselect("Must evidence skills", sorted({s for r in ok for s in r["matched_required"] + r["missing_required"]}))
            sort_by = f4.selectbox("Sort by", ["rank", "score", "required_coverage", "experience_years", "baseline_score", "completeness"])
            view = df[(df.score >= min_score) & (df.required_coverage >= min_cov)]
            view = view[[all(n in m for n in need) for m in view.matched_required]] if need else view
            view = view.sort_values(sort_by, ascending=(sort_by == "rank"))
            show = view.assign(matched_required=view.matched_required.str.join(", "), missing_required=view.missing_required.str.join(", "), possibly_missed=view.possibly_missed.str.join(", "))
            cols = ["rank", "candidate", "score", "band", "required_coverage", "missing_required", "possibly_missed", "experience_years", "baseline_score", "confidence", "near_tie_with_next"] + ([] if anonymize else ["file"])
            st.dataframe(show[cols], hide_index=True, use_container_width=True)
            st.caption("Ranking is deterministic (score, then required coverage, then file name). 'near_tie_with_next' means scores differ by < 3 points: review those manually. 'possibly_missed' lists skills not found in poorly parsed resumes.")
            st.bar_chart(view.set_index("candidate")["score"])
            st.markdown("#### Compare candidates")
            pick = st.multiselect("Choose 2–4 candidates", list(view.candidate), default=list(view.candidate[:2]), max_selections=4)
            if pick:
                dims = [c for c in df.columns if c.startswith("dim_")]
                cmp_ = view[view.candidate.isin(pick)].set_index("candidate")[["score", "required_coverage", "experience_years", *dims]].T
                cmp_.index = [i.replace("dim_", "dimension: ") for i in cmp_.index]
                st.dataframe(cmp_, use_container_width=True)
            st.markdown("#### Candidate details")
            by_name = {r["candidate"]: r["file"] for r in ok}
            by_file = {a.filename: a for a in res}
            order = {a.filename: a for a in res if a.ok}
            ranked_files = [a.filename for a in sorted(order.values(), key=lambda a: (-round(a.match.score, 2), -a.match.required_coverage, a.filename.lower()))]
            real_file = dict(zip([r["candidate"] for r in ok], ranked_files))
            sel = st.selectbox("Open a candidate", list(view.candidate))
            show_analysis(by_file[real_file[sel]], "cand")
            st.markdown("#### Export")
            e1, e2, e3 = st.columns(3)
            e1.download_button("Ranking CSV", ranking_csv(rows), "ranking.csv", "text/csv")
            e2.download_button("Ranking report (HTML)", ranking_html(rows, jd.title, anonymize), "ranking.html", "text/html")
            e3.download_button("Full JSON", ranking_json(rows, res, jd.title, anonymize), "ranking.json", "application/json")

with tab_eval:
    res_dir = ROOT / "evaluation" / "results"
    st.warning("All evaluation data are **synthetic**. They show how the pipeline behaves, not accuracy on real resumes. "
               "The held-out test is the most trustworthy number here; the development and validation sets were used to build the system.")
    parts = [("Held-out test", "HELDOUT.md"), ("Validation (used for diagnosis)", "VALIDATION.md"), ("Development", "DEV_RESULTS.md"),
             ("Config selection", "config_selection.md"), ("Round 1 (first run)", "HELDOUT_round1.md")]
    present = [(t, f) for t, f in parts if (res_dir / f).exists()]
    if present:
        for tab, (title, fname) in zip(st.tabs([t for t, _ in present]), present):
            with tab:
                st.markdown((res_dir / fname).read_text())
    else:
        st.info("No evaluation results yet. Run `python -m evaluation.run_heldout`.")

with tab_about:
    st.markdown((ROOT / "docs" / "METHODOLOGY.md").read_text() if (ROOT / "docs" / "METHODOLOGY.md").exists() else "See README.")
