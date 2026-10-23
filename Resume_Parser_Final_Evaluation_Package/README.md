# Resume Parser — Final Evaluation Package Only

This ZIP contains evaluation artifacts only. It does NOT contain or replace your main project folder or Streamlit app.

## Files
- `Resume_Parser_Evaluation_v3.ipynb` — compares semantic similarity, TF-IDF, hybrid scoring, and a full pipeline including skill coverage.
- `human_review_resume_job_pairs_v3.csv` — 224 fictional resume–job pairs across 16 profiles and 14 job roles. Proposed labels can be reviewed in `reviewer_label` and `reviewer_notes`.
- `README.md` — these instructions.

## Safe usage
1. Extract this ZIP into a NEW folder, for example `Resume_Parser_Evaluation`.
2. Do not overwrite, move, or replace your existing Intelligent Resume Parser project folder.
3. Open the notebook in Jupyter.
4. Keep the CSV in the same folder as the notebook.
5. Run the notebook cells from top to bottom.

The first run needs internet access to download Sentence Transformer model weights.

## Label status and integrity
The CSV contains proposed starter labels, not independently human-validated ground truth. To use human-reviewed labels:
1. Fill `reviewer_label` with 0 (not relevant), 1 (partially relevant), or 2 (strongly relevant).
2. Add optional notes to `reviewer_notes`.
3. Save the CSV.
4. In the notebook set `USE_REVIEWER_LABELS = True` and rerun from the top.

Do not report results from proposed synthetic labels as real-world hiring accuracy. The benchmark tests the evaluation workflow; it is not evidence of hiring performance.
