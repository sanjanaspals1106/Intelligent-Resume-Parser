# Methodology and interpretation

## Pipeline
`file → extraction → parser → job parser → hybrid matcher → assessment + recommendations → reports`

1. **Extraction** (`resume_intel/extraction.py`): PDF via PyMuPDF (reading order sorted), DOCX via python-docx (paragraphs, tables, headers, hyperlink targets). Pages with no text layer are OCR'd with Tesseract when installed (max 6 pages). Corrupt, empty, encrypted, wrongly-typed or text-less files raise a clear `ExtractionError`; in batches the failing file is reported and processing continues.
2. **Parsing** (`parser.py`): rule-based section detection, then per-section parsers for contact details, experience, education, projects, certifications and skills. Skills are normalised through a transparent taxonomy (`skills.py`, ~118 skills with aliases, categories, families and implications). Every entry has a `confidence` (high/medium/low); skill records state *where* each skill appears and keep evidence lines; fields not found are left empty and generate warnings, never guesses (the one exception, deriving a name from an e-mail address, is flagged low confidence).
3. **Job description parsing** (`jobs.py`): required vs preferred skills from headings ("Requirements", "Nice to have"…) and inline cues ("preferred", "a plus", "must"); "X or Y" alternatives collapse into one requirement; skills only mentioned in responsibilities are treated as *preferred* when the JD has an explicit requirements list (and the basis is recorded); minimum years, degree level/field and certifications are extracted; short capitalised terms outside the taxonomy are matched literally.
4. **Matching** (`matching.py`): see below.
5. **Assessment / recommendations** (`assessment.py`, `recommendations.py`): evidence-backed strengths, gaps, uncertainties and rule-based recommendations; no external AI service is used.

## Compatibility score (0-100)
Weighted sum of seven dimensions, each scored 0-1. Dimensions that do not apply to the job (no degree stated, no certification mentioned) or to the resume (no projects listed) are dropped and the remaining weights renormalised, so the breakdown always sums to the score.

| Dimension | Weight | How it is computed |
|---|---|---|
| Skill alignment | 0.30 | Per requirement: **1.0** evidenced in experience/projects/summary/certifications; **0.9** implied (PyTorch → deep learning); **0.5** listed only in a skills list (claimed, not demonstrated); **0.4** a related-family skill is present; **0** missing. Required weight 1.0, preferred 0.5. "X or Y" alternatives take the best option. |
| Experience relevance | 0.25 | Each role: ½ semantic match to the job's requirement lines + ½ share of job skills evidenced in that role. Score = ½ best role + ½ mean of top-3; if the job states a minimum, 75% relevance + 25% years-fit (internships count 50%). |
| Semantic similarity | 0.20 | Each job requirement line is matched to its most similar resume evidence line (all-MiniLM-L6-v2 cosine, rescaled 0.20→0, 0.60→1); weighted mean. |
| Project relevance | 0.10 | Same as experience, over projects. **Not applied if the resume lists no projects** (many professionals list none). |
| Education | 0.05 | Degree level vs requirement (1 / 0.5 one level below / 0.2 lower), 70/30 with field match when a field is named. "Or equivalent experience" floors at 0.5. |
| Certifications | 0.05 | Overlap between certifications named in the job and those on the resume (exact 1.0, related 0.5). |
| Lexical | 0.05 | TF-IDF (1-2 grams) cosine, saturating at 0.35. Deliberately small: shared wording is not shared meaning. |

**Bands:** ≥75 strong, ≥55 good, ≥35 partial, otherwise weak. These label *estimated content relevance*; they are not probabilities. The score is not a hiring probability, not an ATS pass rate, and says nothing about any commercial ATS.

**Baseline.** `resume_intel/baseline.py` keeps the original method (dictionary skill coverage ×0.6 + TF-IDF similarity ×0.4); it is reported next to every hybrid score and in every evaluation. One original bug was fixed: doubled backslashes in `app.py`'s regexes (`\\s` matched a literal backslash, so whitespace was never normalised).

## How the final configuration was chosen (and what did not work)
**Diagnosis.** The first evaluation showed that *removing* the skill dimension raised nDCG@10 (0.881 vs 0.841). Investigating (per-dimension correlation with the labels and mean dimension scores by candidate type) showed that (i) the skill dimension was the most gameable: keyword-stuffed resumes scored 0.80 on it against a true grade of 0.5 because list-only skills received 0.85 credit, and (ii) in that development set "paraphrased" candidates contained *no* skill keywords at all, an extreme that penalised the skill dimension unrealistically. Experience and semantic dimensions, which require evidence in context, tracked the labels better. Across three development corpora the "no skills" advantage shrank to about +0.01 nDCG@10, i.e. partly seed noise.

**Changes made (principled, small).** List-only credit 0.85 → 0.5; related-skill credit 0.5 → 0.4 so an exact-but-undemonstrated claim outranks a merely related skill; weights shifted toward evidence-bearing dimensions (skills 0.35→0.30, experience 0.20→0.25, semantic 0.15→0.20, education/certifications 0.08/0.07→0.05/0.05); projects not applied when absent. Only six configurations were compared (`evaluation/select_config.py`, three development corpora), not a grid search; differences between the top configurations were within noise, so the one that kept skills the largest single weight (interpretable, and keyword evidence matters on real resumes) was chosen.

**Tried and rejected.** Per-skill semantic credit (match "experience with X" against resume lines to credit paraphrased skills): the correct skill was retrieved at rank 1 only 44-49% of the time, too noisy to trust, so paraphrase handling stays in the aggregate semantic and experience dimensions.

**What the held-out test then showed.** Development gains did not transfer fully (see `evaluation/results/HELDOUT.md`): on the untouched test the final hybrid is ahead of the original baseline (+0.068 nDCG@10) but the 95% interval over 10 jobs includes zero, TF-IDF alone scored about as well, and the initial configuration was slightly better than the final one. The final configuration was kept rather than swapped after seeing test results, because choosing on the test set would invalidate it. Honest reading: on synthetic data the hybrid is *not demonstrably better at ranking* than a simple lexical baseline; its demonstrated advantages are behavioural (keyword-stuffed resumes pushed down, paraphrased/near-miss/ambiguous candidates ranked higher) and explanatory (every score is decomposed with evidence).

**Trade-offs worth knowing.** (1) Paraphrase vs keyword evidence: weighting keyword-based skill alignment more helps resumes that use standard terms; weighting semantic similarity more helps paraphrased ones but admits more false positives. (2) Absent evidence scores 0 on experience (flagged as possibly an extraction gap) but projects/education/certification are skipped when not applicable. (3) Project-only candidates (limited professional experience) are ranked slightly lower than the baseline did on the test set; this remains a known weakness.

## Missing vs "could not extract"
A missing requirement is marked `uncertain` when the resume parse quality is below 0.6 or no skills section was found. `parse_quality` is a heuristic of how much of the document was understood (contact info, expected sections found, share of confidently parsed entries, OCR penalty). Uncertain gaps are shown under "Uncertainties", not "Gaps"; an undetected education section yields "may be an extraction gap" rather than a failed requirement; undeterminable years are not scored.

## Fairness safeguards and audit
Scoring reads only job-relevant content: skills, experience/project/summary text with dates removed, degree level and field, certifications. **Excluded from every scoring path:** name, e-mail, phone, links, institution names, education dates and grades, hobbies/interests, references, spoken languages. Years of experience are used only when the job states a minimum, and employment gaps are not penalised (months are a union of stated ranges).

`tests/test_fairness.py` asserts exact score invariance to: candidate name (12 name pairs from different cultures), contact details and links, institution (e.g. a community college vs MIT), education dates and grades, an 18-month employment gap, and added interests/references. It also documents, as a test, what *does* change the score by design: relevant wording (deleting "k8s" lowers the Kubernetes credit).

**Residual risks the tests cannot remove:** wording and style differences (a candidate who describes the same skill differently scores differently, mitigated but not removed by the semantic dimension); parsing quality varies by layout, so unusual layouts are disadvantaged (flagged through `confidence` and "possibly missed" lists, not silently); proxies inside free text (employer names, volunteer roles, location words) are not stripped; the skill taxonomy covers mostly tech/data roles; and relevance labels in any evaluation reflect their authors' judgement. Rankings are decision support for a human reviewer, never an automated screen.

## Privacy and data handling
* All processing is local and in memory. The code never writes resumes, extracted text or reports to disk; downloads are generated in memory and sent only to the user's browser.
* Streamlit usage statistics are disabled (`.streamlit/config.toml`); the Hugging Face hub is not contacted unless `RESUME_INTEL_ALLOW_DOWNLOAD=1` is set to fetch the sentence-embedding model once.
* Uploaded files are held in server memory (Streamlit upload + a small `st.cache_data` of recent analyses) until the session ends or the user presses "Clear uploaded data from memory" / stops the app. Blind-review mode removes names, e-mail, phone, links, file names and raw text from tables, reports and JSON/CSV exports (evidence quotations taken from the resume body are still included).
* Error messages contain a short reason and file name, never resume content.
* The repository contains no real resumes: every sample and test fixture is fictional and generated by code (`evaluation/synthetic.py`, `evaluation/heldout.py`).


## Ranking
Sorted by score, then required-skill coverage, then file name (deterministic). Scores within 3 points are flagged as a near tie. Failed files are listed with their error, never silently dropped.

## Recommendation kinds
`improve_existing` (reword/reorder what exists), `add_if_true` (surface genuine omitted information), `develop` (a real, evidenced gap), `restructure` (parseability). Templates use `[placeholders]`; the system never creates skills, outcomes or numbers, and a test asserts templates contain no digits.

## Evaluation
Three separated tiers (development, validation, held-out test) with the protocol, labelling rubric and caveats in [`evaluation/README.md`](../evaluation/README.md); results in `evaluation/results/`.
