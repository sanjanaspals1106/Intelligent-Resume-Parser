"""The original project's matcher, preserved verbatim in behaviour as the comparison baseline.

Original method: curated skill dictionary -> skill coverage of the job's skills; TF-IDF
(1-2 grams) cosine; score = 0.6 * coverage + 0.4 * similarity (similarity alone when the job
mentions no dictionary skills). Only change: whitespace/regex escapes that were doubled in the
original ``app.py`` (``\\\\s`` matched a literal backslash) are fixed.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASELINE_SKILLS = [
    "python", "java", "c++", "c", "sql", "javascript", "typescript",
    "machine learning", "deep learning", "natural language processing", "nlp",
    "generative ai", "large language models", "llm", "prompt engineering",
    "computer vision", "pytorch", "tensorflow", "scikit-learn", "pandas",
    "numpy", "data analysis", "data visualization", "statistics",
    "feature engineering", "model evaluation", "classification", "regression",
    "clustering", "recommendation systems", "transformers", "hugging face",
    "langchain", "rag", "faiss", "vector databases", "fastapi", "flask",
    "streamlit", "rest api", "html", "css", "react", "node.js", "express",
    "git", "github", "postgresql", "mysql", "mongodb", "docker", "aws",
    "azure", "gcp", "data structures", "algorithms", "excel", "power bi",
    "tableau", "etl", "data preprocessing",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s@:/_-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_skills(text: str) -> list[str]:
    normalized = clean_text(text)
    found = []
    for skill in BASELINE_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(clean_text(skill)) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(set(found), key=str.lower)


def baseline_match(resume_text: str, job_text: str) -> dict:
    resume_skills, job_skills = find_skills(resume_text), find_skills(job_text)
    matched = sorted(set(resume_skills) & set(job_skills), key=str.lower)
    missing = sorted(set(job_skills) - set(resume_skills), key=str.lower)
    coverage = len(matched) / len(set(job_skills)) * 100 if job_skills else 0.0
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
        m = vec.fit_transform([clean_text(resume_text), clean_text(job_text)])
        similarity = float(cosine_similarity(m[0:1], m[1:2])[0, 0]) * 100
    except ValueError:  # empty vocabulary
        similarity = 0.0
    score = 0.6 * coverage + 0.4 * similarity if job_skills else similarity
    return {
        "score": round(score, 2), "skill_coverage": round(coverage, 2), "text_similarity": round(similarity, 2),
        "resume_skills": resume_skills, "job_skills": job_skills, "matched": matched, "missing": missing,
    }
