# Evaluation

All data here are **synthetic and fictional**. No real resumes are bundled (they are personal data), so nothing in this folder demonstrates real-world accuracy.

## Protocol: three tiers, kept separate
| Tier | Generator / seed | Used for | Reported in |
|---|---|---|---|
| **Development** | `synthetic.py`, seeds 7 (and 21, 33 for configuration choice); 66 resumes × 8 jobs | Building the parser; choosing the scoring configuration | `results/DEV_RESULTS.md`, `results/config_selection.md` |
| **Validation** (round 1 of the held-out generator) | `heldout.py`, seed 20240611, layouts L1-L4; 140 × 10 | Failure analysis after its **first run** (kept as-is in `HELDOUT_round1.md`). Parser fixes made afterwards were informed by it, so its later numbers are optimistic | `results/VALIDATION.md`, `results/HELDOUT_round1.md` |
| **Held-out TEST** (round 2) | `heldout.py`, seed 20250217, layouts L1-L6 (adds a two-column PDF and a decorated-heading layout); 140 × 10 | Final numbers. Candidates, layouts and phrasing unseen during development; evaluated **once** on the frozen code (system hash recorded in the file); no changes to the parser or scoring after seeing it | `results/HELDOUT.md` |

Honest caveats about the protocol: the same person wrote the system and every generator and label rubric; the test uses the same job-description family and ability world as validation (new seed and layouts, not a new domain); the L5/L6 layouts and the two-column extraction fix were written *before* the round-2 run but after round-1 diagnosis. Round 1 itself had a set-ordering nondeterminism in the generator (found afterwards, fixed; documented in its result file).

## What is measured
* **Extraction:** micro P/R/F1 per field (name, e-mail, phone, skills, titles, organisations, institutions, degree level, projects, certifications) vs generator ground truth, overall and for the "ambiguous" resumes (no headings, missing/contradictory dates). Skills use the taxonomy's canonical names, which flatters the system, and tools outside the taxonomy (JAX, Flink…) are expected misses.
* **Ranking:** nDCG@5/10, MAP, P@5, MRR, Spearman vs graded relevance (0-3) for: the original baseline, TF-IDF only, skills only, the first (v1) hybrid configuration, the final hybrid with each encoder, and a no-skills ablation; bootstrap 95% intervals over jobs for paired differences; mean rank by candidate archetype.
* **Performance and robustness** (development run): seconds per resume, failure rate, nine malformed-input cases.

## Held-out labelling rubric (`heldout.grade`)
Each candidate holds *abilities* (not taxonomy skills) at a depth: professional (weight 1.0), project (0.8) or coursework (0.4). Score = 0.6·essential + 0.3·important + 0.1·nice coverage; grade 3 ≥ 0.8, 2 ≥ 0.55, 1 ≥ 0.3, else 0. An essential ability that is entirely absent caps the grade at 1 (a near-miss is never "strong"); a coursework-only essential caps it at 2. Claimed-but-not-held skills (keyword stuffing) count for nothing. Short experience lowers the grade by one only when essentials are not shown professionally or through projects. Hard cases: paraphrased abilities, project-strong candidates with ≤1 year experience, near-misses, stuffing, transferable tooling (some outside the taxonomy), and ambiguous/contradictory/missing information.

## Commands
```bash
python -m evaluation.run                 # development set -> results/DEV_RESULTS.md
python -m evaluation.select_config       # development-only configuration comparison
python -m evaluation.run_heldout --validation   # round-1 data (validation)
python -m evaluation.run_heldout         # held-out TEST (re-running is fine; do not tune on it)
python -m evaluation.make_samples        # regenerate samples/resumes/
python -m evaluation.run --real my_data.json    # your own labelled data
```
Real-data format (paths relative to the JSON file; keep such data out of the repository):
```json
[{"job_description": "…full text…", "resumes": [{"file": "resumes/a.pdf", "relevance": 3}, {"file": "resumes/b.docx", "relevance": 0}]}]
```
`--real` prints ranking metrics for the baseline and the hybrid. **Until you run it on real labelled data, real-world performance remains unmeasured.**
