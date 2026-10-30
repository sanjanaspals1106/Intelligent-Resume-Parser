# VALIDATION set evaluation (synthetic) — used for failure analysis; NOT a test result

Generated 2026-10-30T19:58:09 · dataset `311a3b0bd9f40b2d` · system hash `7fe704fdfa6147c2` · generator hash `960f757762a9cfc2` · encoder all-MiniLM-L6-v2 · `python -m evaluation.run_heldout`

> **VALIDATION set.** This is the round-1 held-out data. After its first run (see `HELDOUT_round1.md`) it was used to diagnose failures, so the parser fixes made afterwards were informed by it. These numbers are therefore optimistic and must not be read as test results; the untouched test is `HELDOUT.md`.

## Dataset
- 140 candidates × 10 job descriptions (1400 graded pairs). Each candidate was written for one target job but is graded against every job.
- Archetypes (per job: exact 2, paraphrased 2, project-strong 2, near-miss 2, stuffed 1, transferable 2, ambiguous 2, unrelated 1): {'ambiguous': 20, 'exact': 20, 'near_miss': 20, 'paraphrased': 20, 'project': 20, 'stuffed': 10, 'transferable': 20, 'unrelated': 10}
- Layouts (L1-L4: varied headings, bullets, date formats; L3 DOCX; L4 unconventional headings like 'What I have done'; L5 two-column PDF; L6 decorated headings with company-first headers and dates on their own line): {'L1': 35, 'L2': 35, 'L3': 35, 'L4': 35}
- Grade distribution over all pairs (0-3): {0: 1030, 1: 267, 2: 32, 3: 71}
- Labels: a written rubric over latent *ability* facts (essential 0.6 / important 0.3 / nice 0.1; depth weights professional 1.0, project 0.8, coursework 0.4; an absent essential caps the grade at 1; keyword claims without the ability count for nothing; short experience lowers the grade only if essentials are not shown professionally or via projects). See `heldout.py`.
- Hard cases: paraphrased abilities, project-strong candidates with ≤1 year experience, near-misses missing one essential, keyword stuffing, transferable tooling (some tools such as JAX, Flink, Nomad are deliberately outside the skill taxonomy), and ambiguous resumes (no headings, missing dates, end-before-start dates).

## Ranking

| System | nDCG@5 | nDCG@10 | MAP | P@5 | MRR | Spearman |
|---|---|---|---|---|---|---|
| Original baseline (dictionary skills + TF-IDF) | 0.620 | 0.655 | 0.591 | 0.600 | 0.870 | 0.572 |
| TF-IDF only | 0.649 | 0.693 | 0.667 | 0.720 | 0.833 | 0.517 |
| Skill alignment only | 0.579 | 0.649 | 0.653 | 0.600 | 0.814 | 0.669 |
| Hybrid v1 (initial config, before development tuning) | 0.736 | 0.749 | 0.718 | 0.740 | 0.817 | 0.664 |
| FINAL hybrid, MiniLM encoder | 0.727 | 0.765 | 0.699 | 0.700 | 0.833 | 0.684 |
| FINAL hybrid, fallback concept encoder | 0.703 | 0.726 | 0.683 | 0.740 | 0.750 | 0.640 |
| Final without skills dimension (ablation) | 0.689 | 0.755 | 0.643 | 0.640 | 0.817 | 0.674 |

Paired difference vs the original baseline over the 10 jobs (bootstrap 95% interval over jobs):

| System | Δ nDCG@10 [95% CI] | Δ MAP [95% CI] | jobs improved / worse (nDCG@10) |
|---|---|---|---|
| TF-IDF only | +0.038 [-0.055, +0.133] | +0.077 [-0.013, +0.182] | 6 / 4 |
| Skill alignment only | -0.006 [-0.142, +0.130] | +0.062 [-0.082, +0.204] | 4 / 6 |
| Hybrid v1 (initial config, before development tuning) | +0.093 [-0.004, +0.201] | +0.127 [-0.001, +0.255] | 6 / 4 |
| FINAL hybrid, MiniLM encoder | +0.109 [+0.002, +0.218] | +0.108 [+0.000, +0.223] | 8 / 2 |
| FINAL hybrid, fallback concept encoder | +0.070 [-0.045, +0.188] | +0.093 [-0.023, +0.208] | 7 / 3 |
| Final without skills dimension (ablation) | +0.100 [+0.005, +0.207] | +0.052 [-0.056, +0.166] | 6 / 4 |

Mean rank (1 = best, among all 140 candidates) of each archetype for its own target job:

| Archetype | n | mean true grade | Original baseline rank | Final hybrid rank |
|---|---|---|---|---|
| ambiguous | 20 | 2.2 | 13.6 | 6 |
| exact | 20 | 3 | 6.7 | 3.3 |
| near_miss | 20 | 1 | 12.2 | 6.0 |
| paraphrased | 20 | 3 | 25.1 | 17.9 |
| project | 20 | 2 | 16.2 | 12.2 |
| stuffed | 10 | 0.1 | 21.3 | 21.7 |
| transferable | 20 | 3 | 14.3 | 9.7 |
| unrelated | 10 | 0 | 117.3 | 118 |

## Extraction

| Field | Precision | Recall | F1 | gold |
|---|---|---|---|---|
| name | 1.000 | 1.000 | 1.000 | 140 |
| email | 1.000 | 1.000 | 1.000 | 140 |
| phone | 1.000 | 1.000 | 1.000 | 140 |
| skills | 0.813 | 0.934 | 0.869 | 558 |
| exp_title | 0.995 | 0.995 | 0.995 | 217 |
| exp_org | 0.995 | 0.995 | 0.995 | 217 |
| edu_institution | 1.000 | 1.000 | 1.000 | 140 |
| edu_level | 1.000 | 1.000 | 1.000 | 140 |
| project_name | 1.000 | 0.705 | 0.827 | 88 |
| cert_name | 0.765 | 0.975 | 0.857 | 40 |

Overall excluding skills: P=0.989 R=0.977 F1=0.983; non-ambiguous resumes F1=0.980; ambiguous resumes (no headings / missing or contradictory dates) F1=0.986.
Gold skills outside the taxonomy that were still recovered: 0/37 (expected ≈0: they are unknown to the system).

## Interpretation limits
- 10 jobs and 140 candidates, all synthetic; confidence intervals over jobs are wide.
- Labels come from the generator's latent facts and a rubric written by the same author as the system; real recruiters may disagree, especially on transferable skills and project-only candidates.
- Extraction gold for skills uses the taxonomy's canonical names, which favours the system; names, contacts, titles, organisations, institutions and certifications are the more informative extraction fields.
