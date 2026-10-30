# Scoring configuration selection (DEVELOPMENT data only)

Three development corpora (seeds (7, 21, 33)), 8 jobs each = 24 ranking problems; encoder all-MiniLM-L6-v2.
The held-out set was not used. Differences of ~0.01 are within noise at this size.
All rows use the final defaults for the two smaller changes made after the first evaluation (related-skill credit 0.4 and 'projects not applicable when none listed'); each was checked separately on these corpora and was neutral (±0.003 nDCG@10).

| Configuration | nDCG@10 | MAP | P@5 | Spearman |
|---|---|---|---|---|
| C0 weights v1, list-only credit 0.85 | 0.821 | 0.784 | 0.825 | 0.588 |
| C1 list-only credit 0.5 | 0.831 | 0.795 | 0.842 | 0.597 |
| C2 list-only credit 0.0 | 0.818 | 0.786 | 0.825 | 0.604 |
| C3 SELECTED: weights v2, list-only credit 0.5 | 0.832 | 0.800 | 0.842 | 0.613 |
| C4 evidence-heavy: skills .25 / exp .25 / semantic .25 | 0.835 | 0.806 | 0.833 | 0.622 |
| C5 ablation: no skills dimension | 0.820 | 0.794 | 0.767 | 0.640 |
| Original baseline | 0.752 | 0.689 | 0.767 | 0.483 |
