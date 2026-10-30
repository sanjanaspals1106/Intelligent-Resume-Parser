"""Reproducible evaluation: extraction P/R/F1, ranking metrics (baseline vs hybrid), performance, robustness.

    python -m evaluation.run                # full synthetic evaluation, writes evaluation/results/
    python -m evaluation.run --real FILE    # rank-evaluate your own labelled data (see README)

All data used by the default run is SYNTHETIC (see evaluation/synthetic.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics as st
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TODAY = date(2025, 6, 1)
SEED = 7

from resume_intel import skills as sk  # noqa: E402
from resume_intel.baseline import baseline_match  # noqa: E402
from resume_intel.extraction import ExtractionError, extract_text, ocr_available  # noqa: E402
from resume_intel.jobs import parse_job_description  # noqa: E402
from resume_intel.matching import DEFAULT_WEIGHTS, match  # noqa: E402
from resume_intel.parser import parse_resume  # noqa: E402
from resume_intel.pipeline import analyze_file  # noqa: E402
from resume_intel.semantic import ConceptEncoder, get_encoder  # noqa: E402

from . import synthetic as syn  # noqa: E402
from .metrics import count_matches, items_match, norm, prf, ranking_metrics  # noqa: E402


def _links(v):
    return re.sub(r"^(https?://)?(www\.)?", "", v or "").rstrip("/").lower()


import re  # noqa: E402


def predict_fields(r):
    return {
        "name": [r.name] if r.name else [], "email": [r.email] if r.email else [], "phone": ["".join(c for c in r.phone if c.isdigit())] if r.phone else [],
        "linkedin": [_links(r.links.get("linkedin"))] if r.links.get("linkedin") else [], "github": [_links(r.links.get("github"))] if r.links.get("github") else [],
        "skills": sorted(r.skill_names),
        "edu_institution": [e.institution for e in r.education if e.institution], "edu_level": [f"L{e.level}" for e in r.education if e.level],
        "exp_title": [e.title for e in r.experience if e.title], "exp_org": [e.organization for e in r.experience if e.organization],
        "exp_dates": [f"{e.dates.start}>{e.dates.end}" for e in r.experience if e.dates.start and e.dates.end],
        "project_name": [p.name for p in r.projects if p.name], "cert_name": [c.name for c in r.certifications],
    }


def gold_fields(p):
    t = p.truth()
    t["linkedin"], t["github"] = [_links(t["linkedin"])], [_links(t["github"])]
    for k in ("name", "email", "phone"):
        t[k] = [t[k]]
    t["edu_level"] = [f"L{x}" for x in t["edu_level"]]
    return t


def eval_extraction(personas, texts, label):
    """texts: pid -> (ParsedResume). Micro-averaged P/R/F1 per field."""
    agg = {}
    for p in personas:
        pred, gold = predict_fields(texts[p.pid]), gold_fields(p)
        for f, g in gold.items():
            if f == "skills" and p.kind == "paraphrased":
                continue  # skills are deliberately not named in these resumes
            if f == "exp_dates":
                tp = sum(1 for x in pred[f] if x in g)
                gl, pl = len(g), len(pred[f])
            elif f in ("email", "phone", "linkedin", "github", "edu_level"):
                tp = sum(1 for x in set(pred[f]) if x in g) if f != "edu_level" else count_matches(pred[f], g)
                gl, pl = len(g), len(pred[f])
            else:
                tp, gl, pl = count_matches(pred[f], g), len(g), len(pred[f])
            a = agg.setdefault(f, [0, 0, 0])
            a[0] += tp; a[1] += pl; a[2] += gl
    return {f: prf(*v) for f, v in agg.items()}


def micro_all(per_field, exclude=("skills",)):
    tp = sum(v["tp"] for k, v in per_field.items() if k not in exclude)
    pr = sum(v["predicted"] for k, v in per_field.items() if k not in exclude)
    gd = sum(v["gold"] for k, v in per_field.items() if k not in exclude)
    return prf(tp, pr, gd)


def tiebreak(pid, jid):
    return hashlib.sha1(f"{pid}|{jid}".encode()).hexdigest()


def run(quick=False):
    t_start = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    personas, jobs = syn.build_corpus(SEED)
    if quick:
        personas = [p for i, p in enumerate(personas) if i % 3 == 0]
    blobs = {}
    for p in personas:
        fn, data = syn.render(p)
        blobs[p.pid] = (fn, data)

    # ------------------------------------------------------------- extraction (born-digital)
    parsed, ext_times = {}, {}
    for p in personas:
        fn, data = blobs[p.pid]
        t0 = time.perf_counter()
        ex = extract_text(data, fn)
        parsed[p.pid] = parse_resume(ex.text, ex.links, ex.method, TODAY)
        ext_times[p.pid] = time.perf_counter() - t0
    extraction = {"n_resumes": len(personas), "overall_excluding_skills": micro_all(eval_extraction(personas, parsed, "all")), "per_field": eval_extraction(personas, parsed, "all"), "by_layout": {}}
    for lay in syn.LAYOUTS:
        sub = [p for p in personas if p.layout == lay]
        pf = eval_extraction(sub, parsed, lay)
        extraction["by_layout"][lay] = {"n": len(sub), "micro_f1_excluding_skills": micro_all(pf)["f1"], "skills_f1": pf["skills"]["f1"] if "skills" in pf else None}
    # skill false positives where no skill keyword was written (paraphrased resumes)
    para = [p for p in personas if p.kind == "paraphrased"]
    extraction["paraphrased_resumes_mean_skills_detected"] = round(st.mean(len(parsed[p.pid].skill_names) for p in para), 2) if para else None

    # ------------------------------------------------------------- OCR subset
    ocr = {"tesseract_available": ocr_available()}
    if ocr["tesseract_available"]:
        sub = [p for p in personas if p.layout == "classic"][:6]
        scans = {}
        t0 = time.perf_counter()
        for p in sub:
            fn, data = syn.render(p, scanned=True)
            ex = extract_text(data, fn)
            scans[p.pid] = parse_resume(ex.text, ex.links, ex.method, TODAY)
        ocr["n"] = len(sub)
        ocr["seconds_per_resume"] = round((time.perf_counter() - t0) / len(sub), 2)
        pf = eval_extraction(sub, scans, "ocr")
        ocr["overall_excluding_skills"] = micro_all(pf)
        ocr["per_field_f1"] = {k: v["f1"] for k, v in pf.items()}
        # same resumes via text layer, for a like-for-like comparison
        ocr["same_resumes_text_layer_f1"] = micro_all(eval_extraction(sub, parsed, "text"))["f1"]
    extraction["ocr_scanned_pdfs"] = ocr

    # ------------------------------------------------------------- matching / ranking
    enc_minilm, enc_concept = None, ConceptEncoder()
    try:
        enc_minilm = get_encoder("minilm")
    except Exception as exc:
        enc_minilm = None
        minilm_err = str(exc)
    jdp = {j["id"]: parse_job_description(j["text"]) for j in jobs}
    zero = {k: 0.0 for k in DEFAULT_WEIGHTS}
    systems = {
        "baseline (original: dict skills + TF-IDF)": None,
        "lexical only (TF-IDF)": (enc_concept, {**zero, "lexical": 1.0}),
        "skills only (taxonomy alignment)": (enc_concept, {**zero, "skills": 1.0}),
        "hybrid, concept-hash fallback encoder": (enc_concept, None),
    }
    if enc_minilm:
        systems["hybrid, MiniLM encoder (default)"] = (enc_minilm, None)
        for k in DEFAULT_WEIGHTS:
            systems[f"  ablation: hybrid MiniLM without {k}"] = (enc_minilm, {**DEFAULT_WEIGHTS, k: 0.0})
        for lab, cats in (("paraphrased", {"paraphrased"}), ("stuffed", {"stuffed"})):
            pass
    per_job = {name: {} for name in systems}
    score_cache = {name: {} for name in systems}
    for j in jobs:
        jd = jdp[j["id"]]
        grades = [syn.grade(p, j) for p in personas]
        if not any(g >= 2 for g in grades):
            continue
        tb = [tiebreak(p.pid, j["id"]) for p in personas]
        for name, cfg in systems.items():
            if cfg is None:
                scores = [baseline_match(parsed[p.pid].text, jd.text)["score"] for p in personas]
            else:
                enc, w = cfg
                scores = [match(parsed[p.pid], jd, enc, w).score for p in personas]
            score_cache[name][j["id"]] = scores
            per_job[name][j["id"]] = ranking_metrics(scores, grades, tb)
    metric_names = ["ndcg@5", "ndcg@10", "map", "p@5", "mrr", "spearman"]
    summary = {}
    for name in systems:
        pj = per_job[name]
        summary[name] = {m: round(st.mean(v[m] for v in pj.values()), 4) for m in metric_names}
        summary[name]["std_across_jobs_ndcg@10"] = round(st.pstdev([v["ndcg@10"] for v in pj.values()]), 4)
    # paired comparison hybrid vs baseline per job
    paired = {}
    base = "baseline (original: dict skills + TF-IDF)"
    for name in systems:
        if name == base or name.startswith("  ablation"):
            continue
        d = [per_job[name][j]["ndcg@10"] - per_job[base][j]["ndcg@10"] for j in per_job[base]]
        paired[name] = {"mean_ndcg@10_gain": round(st.mean(d), 4), "jobs_improved": sum(x > 1e-9 for x in d), "jobs_tied": sum(abs(x) <= 1e-9 for x in d), "jobs_worse": sum(x < -1e-9 for x in d), "n_jobs": len(d)}
    # breakdown by candidate type: mean score separation (hybrid vs baseline) for hard cases
    hard = {}
    main = "hybrid, MiniLM encoder (default)" if enc_minilm else "hybrid, concept-hash fallback encoder"
    for kind in ("paraphrased", "stuffed"):
        idx = [i for i, p in enumerate(personas) if p.kind == kind]
        res = {}
        for name in (base, main):
            ranks = []
            for j in jobs:
                if j["id"] not in score_cache[name]:
                    continue
                scores = score_cache[name][j["id"]]
                grades = [syn.grade(p, j) for p in personas]
                tb = [tiebreak(p.pid, j["id"]) for p in personas]
                order = sorted(range(len(scores)), key=lambda i: (-round(scores[i], 6), tb[i]))
                pos = {i: r + 1 for r, i in enumerate(order)}
                for i in idx:
                    if personas[i].family == j["family"]:
                        ranks.append((pos[i], grades[i]))
            res[name] = {"mean_rank_in_pool": round(st.mean(r for r, _ in ranks), 1), "n_pairs": len(ranks), "mean_true_grade": round(st.mean(g for _, g in ranks), 2)}
        hard[kind] = res
    matching = {"n_resumes": len(personas), "n_jobs": len(per_job[base]), "encoder_main": main, "summary": summary, "paired_vs_baseline": paired, "hard_cases_mean_rank": hard, "per_job": per_job["hybrid, MiniLM encoder (default)" if enc_minilm else "hybrid, concept-hash fallback encoder"] if per_job else {}}

    # ------------------------------------------------------------- performance + robustness
    jd0 = jdp["J1"]
    enc = enc_minilm or enc_concept
    analyze_file(blobs[personas[0].pid][1], blobs[personas[0].pid][0], jd0, enc, TODAY)  # warm-up (model load)
    if hasattr(enc, "_cache"):
        enc._cache.clear()  # time from a cold embedding cache so the figure is not flattered by earlier runs
    times = []
    fails = 0
    for p in personas:
        fn, data = blobs[p.pid]
        a = analyze_file(data, fn, jd0, enc, TODAY)
        times.append(a.seconds)
        fails += (not a.ok)
    import fitz

    bad = {}
    good = blobs[personas[0].pid][1]
    bad["empty file"] = (b"", "empty.pdf")
    bad["random bytes as PDF"] = (b"%PDF-1.4\n" + bytes(range(256)) * 20, "random.pdf")
    bad["truncated PDF"] = (good[: len(good) // 3], "trunc.pdf")
    bad["text file renamed .pdf"] = (b"this is not a pdf, just plain text " * 5, "fake.pdf")
    bad["zip that is not a DOCX"] = (b"PK\x03\x04" + b"\x00" * 100, "fake.docx")
    bad["unsupported extension"] = (b"hello world " * 10, "resume.rtf")
    bad["legacy .doc"] = (b"\xd0\xcf\x11\xe0" + b"\x00" * 200, "old.doc")
    d = fitz.open(); d.new_page().insert_text((50, 80), "Secret resume text " * 5)
    bad["password-protected PDF"] = (d.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u"), "locked.pdf")
    d2 = fitz.open(); d2.new_page()
    bad["blank PDF page"] = (d2.tobytes(), "blank.pdf")
    robustness = {}
    for k, (data, fn) in bad.items():
        try:
            a = analyze_file(data, fn, jd0, enc, TODAY)
            robustness[k] = {"handled_without_exception": True, "reported_as_error": not a.ok, "message": (a.error or f"recovered partial text; parse confidence {a.match.confidence}")[:110]}
        except Exception as exc:  # would be a defect
            robustness[k] = {"handled_without_exception": False, "message": repr(exc)[:110]}
    performance = {
        "machine": f"{platform.platform()} / Python {platform.python_version()}", "encoder": enc.name, "n_resumes": len(times),
        "seconds_per_resume_mean": round(st.mean(times), 3), "median": round(st.median(times), 3), "p95": round(sorted(times)[int(.95 * (len(times) - 1))], 3), "max": round(max(times), 3),
        "valid_resume_failure_rate": round(fails / len(times), 4),
        "robustness_malformed_inputs": robustness,
        "all_malformed_inputs_handled_gracefully": all(v["handled_without_exception"] for v in robustness.values()),
    }
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "seed": SEED, "data": "SYNTHETIC development set (templated resumes, latent-truth labels)", "quick_mode": quick,
           "extraction": extraction, "matching": matching, "performance": performance, "total_seconds": round(time.perf_counter() - t_start, 1)}
    (RESULTS / "dev_results.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "DEV_RESULTS.md").write_text(render_markdown(out))
    return out


def render_markdown(o) -> str:
    e, m, p = o["extraction"], o["matching"], o["performance"]
    L = ["# Development-set evaluation (synthetic)", "",
         "> This is the DEVELOPMENT set: the parser and scoring configuration were developed and selected with it (and two sibling seeds, see `config_selection.md`). Treat it as a regression benchmark. The untouched held-out results are in `HELDOUT.md`.", "",
         f"Generated {o['generated']} · seed {o['seed']} · `python -m evaluation.run`" + (" (quick mode)" if o["quick_mode"] else ""), "",
         "> **All numbers below come from SYNTHETIC data** (templated resumes and job descriptions with labels derived from each persona's latent skill set). They verify that the pipeline works and expose failure modes. They are **not** an estimate of accuracy on real resumes, and the extraction numbers are optimistic because the generator and the parser were written by the same author.", "",
         f"## 1. Resume information extraction ({e['n_resumes']} synthetic resumes; PDF + DOCX; 4 layouts)", "", "Micro-averaged precision / recall / F1 against generator ground truth (case/punctuation-insensitive match).", "",
         "| Field | Precision | Recall | F1 | gold items |", "|---|---|---|---|---|"]
    for f, v in e["per_field"].items():
        L.append(f"| {f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['gold']} |")
    ov = e["overall_excluding_skills"]
    L += ["", f"Overall (all fields except skills): P={ov['precision']:.3f} R={ov['recall']:.3f} F1={ov['f1']:.3f}.",
          "Skills: the generator draws surface forms from the same taxonomy aliases the extractor uses, so the skills score is a regression check, not evidence of real-world skill-extraction accuracy.", "",
          "| Layout | n | micro-F1 (excl. skills) |", "|---|---|---|"] + [f"| {k} | {v['n']} | {v['micro_f1_excluding_skills']:.3f} |" for k, v in e["by_layout"].items()]
    o_ = e["ocr_scanned_pdfs"]
    if o_.get("tesseract_available") and "n" in o_:
        L += ["", f"**Scanned (image-only) PDFs via Tesseract OCR**, {o_['n']} resumes (clean synthetic renders, not real scans): micro-F1 {o_['overall_excluding_skills']['f1']:.3f} vs {o_['same_resumes_text_layer_f1']:.3f} for the same resumes through the text layer; {o_['seconds_per_resume']} s/resume."]
    else:
        L += ["", "Scanned-PDF OCR was not evaluated (Tesseract not installed)."]
    L += ["", f"## 2. Job matching and ranking ({m['n_resumes']} resumes x {m['n_jobs']} jobs, graded relevance 0-3)", "",
          "Relevance labels come from the persona's latent skills (not from the system). Binary relevance for MAP / P@5 / MRR is grade >= 2. Ties are broken by a fixed hash. Metrics are averaged over jobs.", "",
          "| System | nDCG@5 | nDCG@10 | MAP | P@5 | MRR | Spearman |", "|---|---|---|---|---|---|---|"]
    for name, v in m["summary"].items():
        L.append(f"| {name.strip()} | {v['ndcg@5']:.3f} | {v['ndcg@10']:.3f} | {v['map']:.3f} | {v['p@5']:.3f} | {v['mrr']:.3f} | {v['spearman']:.3f} |")
    L += ["", "Paired per-job comparison against the original baseline (nDCG@10):", "", "| System | mean gain | jobs improved / tied / worse |", "|---|---|---|"]
    for name, v in m["paired_vs_baseline"].items():
        L.append(f"| {name} | {v['mean_ndcg@10_gain']:+.3f} | {v['jobs_improved']} / {v['jobs_tied']} / {v['jobs_worse']} (of {v['n_jobs']}) |")
    L += ["", "Hard cases: mean rank (1 = best) of same-family candidates in the pool of " + str(m["n_resumes"]) + ". *Paraphrased* resumes describe real skills without the usual keywords; *stuffed* resumes list keywords with little real experience (true grades are higher / lower respectively). For *paraphrased* a smaller mean rank is better; for *stuffed* a larger mean rank is better (they should not rank high).", "",
          "| Candidate type | system | mean rank | mean true grade | pairs |", "|---|---|---|---|---|"]
    for kind, res in m["hard_cases_mean_rank"].items():
        for name, v in res.items():
            L.append(f"| {kind} | {name} | {v['mean_rank_in_pool']} | {v['mean_true_grade']} | {v['n_pairs']} |")
    L += ["", f"## 3. Performance and robustness ({p['n_resumes']} resumes, {p['machine']})", "",
          f"- Encoder: {p['encoder']}. Time per resume (file -> parse -> match -> assess): mean {p['seconds_per_resume_mean']} s, median {p['median']} s, p95 {p['p95']} s, max {p['max']} s.",
          f"- Failure rate on valid synthetic resumes: {100 * p['valid_resume_failure_rate']:.1f}%.",
          f"- Malformed inputs handled without an exception (clear error message, or low-confidence partial result): **{p['all_malformed_inputs_handled_gracefully']}**", "", "| Malformed input | Rejected with error | Message / outcome |", "|---|---|---|"]
    for k, v in p["robustness_malformed_inputs"].items():
        L.append(f"| {k} | {v.get('reported_as_error')} | {v['message']} |")
    L += ["", "## Caveats", "- Synthetic data only; 8 jobs is too few for statistical significance; per-job paired counts are provided so you can judge consistency.",
          "- The hybrid system and the labels share the author, so favourable comparisons against the baseline should be confirmed on real, human-labelled data (`--real`).", "- Wall-clock times depend on the machine; they were measured after model load with a cleared embedding cache, on short synthetic resumes (real resumes are longer)."]
    return "\n".join(L) + "\n"


def run_real(path: str):
    """Rank-evaluate user-supplied data. JSON: {"job_description": str, "resumes": [{"file": path, "relevance": 0-3}, ...]} or a list of such objects."""
    spec = json.loads(Path(path).read_text())
    spec = spec if isinstance(spec, list) else [spec]
    enc = get_encoder()
    rows = {"baseline": [], "hybrid": []}
    for job in spec:
        jd = parse_job_description(job["job_description"])
        base, hyb, gr = [], [], []
        for r in job["resumes"]:
            fp = Path(path).parent / r["file"]
            a = analyze_file(fp.read_bytes(), fp.name, jd, enc)
            if a.ok:
                base.append(a.match.baseline["score"]); hyb.append(a.match.score); gr.append(r["relevance"])
        tb = [str(i) for i in range(len(gr))]
        rows["baseline"].append(ranking_metrics(base, gr, tb)); rows["hybrid"].append(ranking_metrics(hyb, gr, tb))
    out = {k: {m: round(st.mean(x[m] for x in v), 4) for m in v[0]} for k, v in rows.items()}
    print(json.dumps({"n_jobs": len(spec), **out}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="evaluate a third of the resumes (fast smoke run)")
    ap.add_argument("--real", help="path to a JSON file with labelled real data")
    a = ap.parse_args()
    if a.real:
        run_real(a.real)
    else:
        res = run(a.quick)
        print((RESULTS / "DEV_RESULTS.md").read_text())
