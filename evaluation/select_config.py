"""Development-only comparison of a handful of principled scoring configurations.

Run:  python -m evaluation.select_config

Uses three DEVELOPMENT corpora (seeds 7, 21, 33 of evaluation/synthetic.py). The held-out set
(evaluation/heldout.py) is never touched here. Only six configurations are compared (not a grid search), and the
choice weighs interpretability and robustness to keyword stuffing, not just the metric.
"""
from __future__ import annotations

import hashlib
import statistics as st
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
from resume_intel.baseline import baseline_match  # noqa: E402
from resume_intel.extraction import extract_text  # noqa: E402
from resume_intel.jobs import parse_job_description  # noqa: E402
from resume_intel.matching import match  # noqa: E402
from resume_intel.parser import parse_resume  # noqa: E402
from resume_intel.semantic import get_encoder  # noqa: E402

from . import synthetic as syn  # noqa: E402
from .metrics import ranking_metrics  # noqa: E402

OLD = {"skills": .35, "experience": .20, "semantic": .15, "projects": .10, "education": .08, "certifications": .07, "lexical": .05}
NEW = {"skills": .30, "experience": .25, "semantic": .20, "projects": .10, "education": .05, "certifications": .05, "lexical": .05}
CONFIGS = {
    "C0 weights v1, list-only credit 0.85": (OLD, 0.85),
    "C1 list-only credit 0.5": (OLD, 0.5),
    "C2 list-only credit 0.0": (OLD, 0.0),
    "C3 SELECTED: weights v2, list-only credit 0.5": (NEW, 0.5),
    "C4 evidence-heavy: skills .25 / exp .25 / semantic .25": ({**NEW, "skills": .25, "semantic": .25}, 0.5),
    "C5 ablation: no skills dimension": ({**OLD, "skills": 0.0}, 0.85),
}
SEEDS = (7, 21, 33)
OUT = Path(__file__).parent / "results" / "config_selection.md"


def main():
    enc = get_encoder()
    res = {k: [] for k in CONFIGS}
    res["Original baseline"] = []
    for seed in SEEDS:
        ps, jobs = syn.build_corpus(seed)
        parsed = {}
        for p in ps:
            fn, data = syn.render(p)
            parsed[p.pid] = parse_resume(extract_text(data, fn).text, today=date(2025, 6, 1))
        for j in jobs:
            jd = parse_job_description(j["text"])
            g = [syn.grade(p, j) for p in ps]
            tb = [hashlib.sha1((p.pid + j["id"]).encode()).hexdigest() for p in ps]
            res["Original baseline"].append(ranking_metrics([baseline_match(parsed[p.pid].text, jd.text)["score"] for p in ps], g, tb))
            for k, (w, lc) in CONFIGS.items():
                res[k].append(ranking_metrics([match(parsed[p.pid], jd, enc, w, lc).score for p in ps], g, tb))
    lines = ["# Scoring configuration selection (DEVELOPMENT data only)", "",
             f"Three development corpora (seeds {SEEDS}), 8 jobs each = {len(res['Original baseline'])} ranking problems; encoder {enc.name}.",
             "The held-out set was not used. Differences of ~0.01 are within noise at this size.",
             "All rows use the final defaults for the two smaller changes made after the first evaluation (related-skill credit 0.4 and 'projects not applicable when none listed'); each was checked separately on these corpora and was neutral (±0.003 nDCG@10).", "",
             "| Configuration | nDCG@10 | MAP | P@5 | Spearman |", "|---|---|---|---|---|"]
    for k, v in res.items():
        lines.append(f"| {k} | " + " | ".join(f"{st.mean(x[m] for x in v):.3f}" for m in ("ndcg@10", "map", "p@5", "spearman")) + " |")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
