"""Final evaluation on the HELD-OUT synthetic set (evaluation/heldout.py). Run once per released configuration.

    python -m evaluation.run_heldout

The held-out data were not used to develop the parser, choose scoring weights or select the configuration. The run
records SHA-256 hashes of the system source and of the generated data so the evaluated code is identifiable.
Everything here is SYNTHETIC; nothing is real-world validation.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics as st
import warnings
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
from resume_intel import skills as sk  # noqa: E402
from resume_intel.baseline import baseline_match  # noqa: E402
from resume_intel.extraction import extract_text  # noqa: E402
from resume_intel.jobs import parse_job_description  # noqa: E402
from resume_intel.matching import DEFAULT_WEIGHTS, match  # noqa: E402
from resume_intel.parser import parse_resume  # noqa: E402
from resume_intel.semantic import ConceptEncoder, get_encoder  # noqa: E402

from . import heldout as ho  # noqa: E402
from .metrics import count_matches, norm, prf, ranking_metrics  # noqa: E402
from .select_config import OLD  # noqa: E402

ROOT = Path(__file__).parent
TODAY = date(2025, 6, 1)
METRICS = ["ndcg@5", "ndcg@10", "map", "p@5", "mrr", "spearman"]


def _sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _pred(r):
    return {"name": [r.name] if r.name else [], "email": [r.email] if r.email else [], "phone": ["".join(c for c in r.phone if c.isdigit())] if r.phone else [],
            "skills": sorted(r.skill_names), "exp_title": [e.title for e in r.experience if e.title], "exp_org": [e.organization for e in r.experience if e.organization],
            "edu_institution": [e.institution for e in r.education if e.institution], "edu_level": [f"L{e.level}" for e in r.education if e.level],
            "project_name": [p.name for p in r.projects if p.name], "cert_name": [c.name for c in r.certifications]}


def eval_extraction(cands, parsed, only=None):
    agg, tax = {}, set(sk.SKILLS)
    out_tax = [0, 0]
    for c in cands:
        if only and c.archetype not in only:
            continue
        pred, gold = _pred(parsed[c.cid]), c.truth()
        for f, g in gold.items():
            if f in ("email", "phone", "edu_level"):
                tp = count_matches(pred[f], g) if f == "edu_level" else sum(x in g for x in set(pred[f]))
            else:
                tp = count_matches(pred[f], g)
            a = agg.setdefault(f, [0, 0, 0])
            a[0] += tp; a[1] += len(pred[f]); a[2] += len(g)
        extra = [x for x in c.gold_skills if x not in tax]
        out_tax[0] += sum(x in {norm(p) for p in pred["skills"]} or x in pred["skills"] for x in extra)
        out_tax[1] += len(extra)
    return {f: prf(*v) for f, v in agg.items()}, out_tax


def main(mode="test"):
    """mode "validation": round-1 data (seed ho.SEED, layouts L1-L4), used for failure analysis and fixes.
    mode "test": fresh candidates (seed ho.TEST_SEED, layouts L1-L6); run once on the frozen system."""
    if mode == "validation":
        cands, jobs = ho.build(ho.SEED)
    else:
        cands, jobs = ho.build(ho.TEST_SEED, layouts=("L1", "L2", "L3", "L4", "L5", "L6"))
    J = {j["id"]: j for j in jobs}
    parsed, enc_f = {}, None
    for c in cands:
        fn, data = ho.render_cand(c)
        ex = extract_text(data, fn)
        parsed[c.cid] = parse_resume(ex.text, ex.links, ex.method, TODAY)
    enc = get_encoder()
    fallback = ConceptEncoder()
    zero = {k: 0.0 for k in DEFAULT_WEIGHTS}
    old_related = 0.5

    def run_v1(r, jd):
        sk.RELATED_CREDIT = old_related
        try:
            return match(r, jd, enc, OLD, 0.85, projects_na_if_none=False).score
        finally:
            sk.RELATED_CREDIT = 0.4

    systems = {
        "Original baseline (dictionary skills + TF-IDF)": lambda r, jd: baseline_match(r.text, jd.text)["score"],
        "TF-IDF only": lambda r, jd: match(r, jd, fallback, {**zero, "lexical": 1.0}).score,
        "Skill alignment only": lambda r, jd: match(r, jd, fallback, {**zero, "skills": 1.0}).score,
        "Hybrid v1 (initial config, before development tuning)": run_v1,
        "FINAL hybrid, MiniLM encoder": lambda r, jd: match(r, jd, enc).score,
        "FINAL hybrid, fallback concept encoder": lambda r, jd: match(r, jd, fallback).score,
        "Final without skills dimension (ablation)": lambda r, jd: match(r, jd, enc, {**DEFAULT_WEIGHTS, "skills": 0.0}).score,
    }
    per_job, scores = {k: {} for k in systems}, {k: {} for k in systems}
    for j in jobs:
        jd = parse_job_description(j["text"])
        g = [ho.grade(c, j) for c in cands]
        tb = [hashlib.sha1(f"{c.cid}|{j['id']}".encode()).hexdigest() for c in cands]
        for name, fn in systems.items():
            sc = [fn(parsed[c.cid], jd) for c in cands]
            scores[name][j["id"]] = sc
            per_job[name][j["id"]] = ranking_metrics(sc, g, tb)
    summary = {n: {m: round(st.mean(v[m] for v in pj.values()), 4) for m in METRICS} for n, pj in per_job.items()}
    base = "Original baseline (dictionary skills + TF-IDF)"
    rng = random.Random(0)
    jids = list(per_job[base])
    paired = {}
    for n in systems:
        if n == base:
            continue
        row = {}
        for m in ("ndcg@10", "map"):
            d = [per_job[n][j][m] - per_job[base][j][m] for j in jids]
            boots = sorted(st.mean(rng.choice(d) for _ in d) for _ in range(4000))
            row[m] = {"mean_gain": round(st.mean(d), 4), "ci95": [round(boots[100], 4), round(boots[3899], 4)], "jobs_improved": sum(x > 1e-9 for x in d), "jobs_worse": sum(x < -1e-9 for x in d)}
        paired[n] = row
    # archetype behaviour: mean rank (1 = best) of each archetype among all candidates, for its own target job
    arch_rank = {n: defaultdict(list) for n in ("Original baseline (dictionary skills + TF-IDF)", "FINAL hybrid, MiniLM encoder")}
    arch_grade = defaultdict(list)
    for j in jobs:
        for n in arch_rank:
            sc = scores[n][j["id"]]
            order = sorted(range(len(cands)), key=lambda i: (-round(sc[i], 6), hashlib.sha1(f"{cands[i].cid}|{j['id']}".encode()).hexdigest()))
            pos = {i: r + 1 for r, i in enumerate(order)}
            for i, c in enumerate(cands):
                if c.target == j["id"]:
                    arch_rank[n][c.archetype].append(pos[i])
                    if n.startswith("FINAL"):
                        arch_grade[c.archetype].append(ho.grade(c, j))
    arch = {a: {"n": len(arch_grade[a]), "mean_grade": round(st.mean(arch_grade[a]), 2),
                **{n.split(" ")[0] + "_mean_rank": round(st.mean(arch_rank[n][a]), 1) for n in arch_rank}} for a in sorted(arch_grade)}
    # extraction
    ext_all, out_tax = eval_extraction(cands, parsed)
    ext_amb, _ = eval_extraction(cands, parsed, only={"ambiguous"})
    ext_clean, _ = eval_extraction(cands, parsed, only={"exact", "paraphrased", "project", "near_miss", "transferable"})
    micro = lambda d, ex=("skills",): prf(sum(v["tp"] for k, v in d.items() if k not in ex), sum(v["predicted"] for k, v in d.items() if k not in ex), sum(v["gold"] for k, v in d.items() if k not in ex))
    # label distribution
    dist = defaultdict(int)
    for c in cands:
        for j in jobs:
            dist[ho.grade(c, j)] += 1
    out = {
        "generated": datetime.now().isoformat(timespec="seconds"), "data": f"SYNTHETIC {mode} set", "mode": mode, "seed": ho.SEED if mode == "validation" else ho.TEST_SEED,
        "system_sha256_16": _sha(Path("resume_intel").glob("*.py")), "heldout_generator_sha256_16": _sha([ROOT / "heldout.py"]),
        "dataset_sha256_16": hashlib.sha256("\n".join(f"{c.cid}|{c.layout}|" + "\n".join(f"{k}:{t}" for k, t in c.structure) for c in cands).encode()).hexdigest()[:16],
        "encoder": enc.name, "n_candidates": len(cands), "n_jobs": len(jobs),
        "composition": {"archetypes": {a: sum(c.archetype == a for c in cands) for a in sorted({c.archetype for c in cands})}, "layouts": {l: sum(c.layout == l for c in cands) for l in ("L1", "L2", "L3", "L4", "L5", "L6") if any(c.layout == l for c in cands)},
                        "label_distribution_all_pairs": dict(sorted(dist.items()))},
        "ranking_summary": summary, "ranking_paired_vs_baseline": paired, "per_job_final": per_job["FINAL hybrid, MiniLM encoder"], "archetype_mean_rank": arch,
        "extraction": {"overall_excluding_skills": micro(ext_all), "per_field": ext_all, "ambiguous_only_excluding_skills": micro(ext_amb), "non_ambiguous_excluding_skills": micro(ext_clean),
                       "skills_outside_taxonomy_recovered": f"{out_tax[0]}/{out_tax[1]}"},
    }
    stem = "validation" if mode == "validation" else "heldout"
    (ROOT / "results" / f"{stem}.json").write_text(json.dumps(out, indent=2))
    (ROOT / "results" / f"{stem.upper()}.md").write_text(render(out))
    print(render(out))


def render(o):
    c = o["composition"]
    val = o["mode"] == "validation"
    L = ["# " + ("VALIDATION set evaluation (synthetic) — used for failure analysis; NOT a test result" if val else "Held-out TEST evaluation (synthetic)"), "",
         f"Generated {o['generated']} · dataset `{o['dataset_sha256_16']}` · system hash `{o['system_sha256_16']}` · generator hash `{o['heldout_generator_sha256_16']}` · encoder {o['encoder']} · `python -m evaluation.run_heldout`", "",
         ("> **VALIDATION set.** This is the round-1 held-out data. After its first run (see `HELDOUT_round1.md`) it was used to diagnose failures, so the parser fixes made afterwards were informed by it. These numbers are therefore optimistic and must not be read as test results; the untouched test is `HELDOUT.md`." if val else "> **SYNTHETIC data.** These resumes and job descriptions were generated by `evaluation/heldout.py` with a fresh seed and two additional layouts (L5 two-column PDF, L6 decorated headings) **after the parser and scoring configuration were frozen**, and were evaluated exactly once. They are still synthetic and written by the system's author: the numbers show behaviour on a harder, differently-styled test, **not** real-world accuracy."), "",
         "## Dataset", f"- {o['n_candidates']} candidates × {o['n_jobs']} job descriptions ({o['n_candidates'] * o['n_jobs']} graded pairs). Each candidate was written for one target job but is graded against every job.",
         f"- Archetypes (per job: exact 2, paraphrased 2, project-strong 2, near-miss 2, stuffed 1, transferable 2, ambiguous 2, unrelated 1): {c['archetypes']}",
         f"- Layouts (L1-L4: varied headings, bullets, date formats; L3 DOCX; L4 unconventional headings like 'What I have done'; L5 two-column PDF; L6 decorated headings with company-first headers and dates on their own line): {c['layouts']}",
         f"- Grade distribution over all pairs (0-3): {c['label_distribution_all_pairs']}",
         "- Labels: a written rubric over latent *ability* facts (essential 0.6 / important 0.3 / nice 0.1; depth weights professional 1.0, project 0.8, coursework 0.4; an absent essential caps the grade at 1; keyword claims without the ability count for nothing; short experience lowers the grade only if essentials are not shown professionally or via projects). See `heldout.py`.",
         "- Hard cases: paraphrased abilities, project-strong candidates with ≤1 year experience, near-misses missing one essential, keyword stuffing, transferable tooling (some tools such as JAX, Flink, Nomad are deliberately outside the skill taxonomy), and ambiguous resumes (no headings, missing dates, end-before-start dates).", "",
         "## Ranking", "", "| System | nDCG@5 | nDCG@10 | MAP | P@5 | MRR | Spearman |", "|---|---|---|---|---|---|---|"]
    for n, v in o["ranking_summary"].items():
        L.append(f"| {n} | " + " | ".join(f"{v[m]:.3f}" for m in METRICS) + " |")
    L += ["", "Paired difference vs the original baseline over the 10 jobs (bootstrap 95% interval over jobs):", "", "| System | Δ nDCG@10 [95% CI] | Δ MAP [95% CI] | jobs improved / worse (nDCG@10) |", "|---|---|---|---|"]
    for n, v in o["ranking_paired_vs_baseline"].items():
        a, b = v["ndcg@10"], v["map"]
        L.append(f"| {n} | {a['mean_gain']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {b['mean_gain']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | {a['jobs_improved']} / {a['jobs_worse']} |")
    L += ["", "Mean rank (1 = best, among all 140 candidates) of each archetype for its own target job:", "", "| Archetype | n | mean true grade | Original baseline rank | Final hybrid rank |", "|---|---|---|---|---|"]
    for a, v in o["archetype_mean_rank"].items():
        L.append(f"| {a} | {v['n']} | {v['mean_grade']} | {v['Original_mean_rank']} | {v['FINAL_mean_rank']} |")
    e = o["extraction"]
    L += ["", "## Extraction", "", "| Field | Precision | Recall | F1 | gold |", "|---|---|---|---|---|"]
    for f, v in e["per_field"].items():
        L.append(f"| {f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['gold']} |")
    ov, am, cl = e["overall_excluding_skills"], e["ambiguous_only_excluding_skills"], e["non_ambiguous_excluding_skills"]
    L += ["", f"Overall excluding skills: P={ov['precision']:.3f} R={ov['recall']:.3f} F1={ov['f1']:.3f}; non-ambiguous resumes F1={cl['f1']:.3f}; ambiguous resumes (no headings / missing or contradictory dates) F1={am['f1']:.3f}.",
          f"Gold skills outside the taxonomy that were still recovered: {e['skills_outside_taxonomy_recovered']} (expected ≈0: they are unknown to the system).", "",
          "## Interpretation limits", "- 10 jobs and 140 candidates, all synthetic; confidence intervals over jobs are wide.", "- Labels come from the generator's latent facts and a rubric written by the same author as the system; real recruiters may disagree, especially on transferable skills and project-only candidates.",
          "- Extraction gold for skills uses the taxonomy's canonical names, which favours the system; names, contacts, titles, organisations, institutions and certifications are the more informative extraction fields."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys

    main("validation" if "--validation" in sys.argv else "test")
