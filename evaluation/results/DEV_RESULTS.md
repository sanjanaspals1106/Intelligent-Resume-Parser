# Development-set evaluation (synthetic)

> This is the DEVELOPMENT set: the parser and scoring configuration were developed and selected with it (and two sibling seeds, see `config_selection.md`). Treat it as a regression benchmark. The untouched held-out results are in `HELDOUT.md`.

Generated 2026-10-30T19:58:38 · seed 7 · `python -m evaluation.run`

> **All numbers below come from SYNTHETIC data** (templated resumes and job descriptions with labels derived from each persona's latent skill set). They verify that the pipeline works and expose failure modes. They are **not** an estimate of accuracy on real resumes, and the extraction numbers are optimistic because the generator and the parser were written by the same author.

## 1. Resume information extraction (66 synthetic resumes; PDF + DOCX; 4 layouts)

Micro-averaged precision / recall / F1 against generator ground truth (case/punctuation-insensitive match).

| Field | Precision | Recall | F1 | gold items |
|---|---|---|---|---|
| name | 1.000 | 1.000 | 1.000 | 66 |
| email | 1.000 | 1.000 | 1.000 | 66 |
| phone | 1.000 | 1.000 | 1.000 | 66 |
| linkedin | 1.000 | 1.000 | 1.000 | 66 |
| github | 1.000 | 1.000 | 1.000 | 66 |
| skills | 0.870 | 0.998 | 0.929 | 416 |
| edu_institution | 1.000 | 1.000 | 1.000 | 66 |
| edu_level | 1.000 | 1.000 | 1.000 | 66 |
| exp_title | 1.000 | 1.000 | 1.000 | 108 |
| exp_org | 1.000 | 1.000 | 1.000 | 108 |
| exp_dates | 1.000 | 1.000 | 1.000 | 108 |
| project_name | 1.000 | 1.000 | 1.000 | 93 |
| cert_name | 1.000 | 1.000 | 1.000 | 58 |

Overall (all fields except skills): P=1.000 R=1.000 F1=1.000.
Skills: the generator draws surface forms from the same taxonomy aliases the extractor uses, so the skills score is a regression check, not evidence of real-world skill-extraction accuracy.

| Layout | n | micro-F1 (excl. skills) |
|---|---|---|
| classic | 16 | 1.000 |
| pipes | 17 | 1.000 |
| docx | 17 | 1.000 |
| minimal | 16 | 1.000 |

**Scanned (image-only) PDFs via Tesseract OCR**, 6 resumes (clean synthetic renders, not real scans): micro-F1 1.000 vs 1.000 for the same resumes through the text layer; 0.87 s/resume.

## 2. Job matching and ranking (66 resumes x 8 jobs, graded relevance 0-3)

Relevance labels come from the persona's latent skills (not from the system). Binary relevance for MAP / P@5 / MRR is grade >= 2. Ties are broken by a fixed hash. Metrics are averaged over jobs.

| System | nDCG@5 | nDCG@10 | MAP | P@5 | MRR | Spearman |
|---|---|---|---|---|---|---|
| baseline (original: dict skills + TF-IDF) | 0.788 | 0.760 | 0.724 | 0.850 | 0.938 | 0.521 |
| lexical only (TF-IDF) | 0.709 | 0.733 | 0.720 | 0.800 | 0.938 | 0.454 |
| skills only (taxonomy alignment) | 0.845 | 0.830 | 0.804 | 0.875 | 1.000 | 0.551 |
| hybrid, concept-hash fallback encoder | 0.864 | 0.874 | 0.828 | 0.900 | 1.000 | 0.597 |
| hybrid, MiniLM encoder (default) | 0.856 | 0.870 | 0.843 | 0.875 | 1.000 | 0.631 |
| ablation: hybrid MiniLM without skills | 0.829 | 0.879 | 0.861 | 0.850 | 1.000 | 0.661 |
| ablation: hybrid MiniLM without experience | 0.845 | 0.862 | 0.829 | 0.875 | 1.000 | 0.592 |
| ablation: hybrid MiniLM without semantic | 0.851 | 0.864 | 0.828 | 0.875 | 1.000 | 0.610 |
| ablation: hybrid MiniLM without projects | 0.853 | 0.860 | 0.840 | 0.875 | 1.000 | 0.631 |
| ablation: hybrid MiniLM without education | 0.856 | 0.870 | 0.843 | 0.875 | 1.000 | 0.631 |
| ablation: hybrid MiniLM without certifications | 0.856 | 0.870 | 0.843 | 0.875 | 1.000 | 0.631 |
| ablation: hybrid MiniLM without lexical | 0.851 | 0.862 | 0.840 | 0.850 | 1.000 | 0.632 |

Paired per-job comparison against the original baseline (nDCG@10):

| System | mean gain | jobs improved / tied / worse |
|---|---|---|
| lexical only (TF-IDF) | -0.028 | 2 / 0 / 6 (of 8) |
| skills only (taxonomy alignment) | +0.070 | 7 / 0 / 1 (of 8) |
| hybrid, concept-hash fallback encoder | +0.114 | 8 / 0 / 0 (of 8) |
| hybrid, MiniLM encoder (default) | +0.110 | 8 / 0 / 0 (of 8) |

Hard cases: mean rank (1 = best) of same-family candidates in the pool of 66. *Paraphrased* resumes describe real skills without the usual keywords; *stuffed* resumes list keywords with little real experience (true grades are higher / lower respectively). For *paraphrased* a smaller mean rank is better; for *stuffed* a larger mean rank is better (they should not rank high).

| Candidate type | system | mean rank | mean true grade | pairs |
|---|---|---|---|---|
| paraphrased | baseline (original: dict skills + TF-IDF) | 26.2 | 2.88 | 16 |
| paraphrased | hybrid, MiniLM encoder (default) | 14.9 | 2.88 | 16 |
| stuffed | baseline (original: dict skills + TF-IDF) | 4.2 | 0.5 | 8 |
| stuffed | hybrid, MiniLM encoder (default) | 9.4 | 0.5 | 8 |

## 3. Performance and robustness (66 resumes, macOS-15.6.1-arm64-arm-64bit-Mach-O / Python 3.13.9)

- Encoder: all-MiniLM-L6-v2. Time per resume (file -> parse -> match -> assess): mean 0.022 s, median 0.021 s, p95 0.033 s, max 0.054 s.
- Failure rate on valid synthetic resumes: 0.0%.
- Malformed inputs handled without an exception (clear error message, or low-confidence partial result): **True**

| Malformed input | Rejected with error | Message / outcome |
|---|---|---|
| empty file | True | File is empty |
| random bytes as PDF | True | Could not open PDF (corrupt or not a PDF): Failed to open stream |
| truncated PDF | False | recovered partial text; parse confidence high |
| text file renamed .pdf | True | Could not open PDF (corrupt or not a PDF): Failed to open stream |
| zip that is not a DOCX | True | File is not a valid DOCX (not a zip container) |
| unsupported extension | True | Unsupported file type (supported: PDF, DOCX) |
| legacy .doc | True | Legacy .doc files are not supported; save the resume as .docx or .pdf |
| password-protected PDF | True | PDF is password-protected |
| blank PDF page | True | Almost no text could be extracted. If this is a scanned PDF, install Tesseract OCR to enable text recognition. |

## Caveats
- Synthetic data only; 8 jobs is too few for statistical significance; per-job paired counts are provided so you can judge consistency.
- The hybrid system and the labels share the author, so favourable comparisons against the baseline should be confirmed on real, human-labelled data (`--real`).
- Wall-clock times depend on the machine; they were measured after model load with a cleared embedding cache, on short synthetic resumes (real resumes are longer).
