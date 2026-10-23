# Intelligent Resume Parser — Advanced Notebook

## Features
- PDF text extraction
- Contact detail extraction
- Curated skill extraction and alias canonicalization
- TF-IDF lexical similarity baseline
- Sentence Transformer semantic similarity (`all-MiniLM-L6-v2`)
- Heuristic section-aware semantic analysis
- Explainable combined score
- Matched/missing skill report
- Optional ranking across multiple job descriptions
- JSON and CSV exports

## Run
1. Open `Intelligent_Resume_Parser_Advanced.ipynb` in Jupyter.
2. Run cells in order.
3. Install dependencies when prompted.
4. Enter the full path to a selectable-text PDF resume.
5. Edit the job description cell and run the analysis.

The first semantic-model run requires internet access to download model weights. The combined score is a heuristic relevance signal, not a calibrated hiring probability.
