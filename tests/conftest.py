import warnings
from datetime import date
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
FIX = Path(__file__).parent / "fixtures"
TODAY = date(2025, 1, 1)

JD_ML = """Machine Learning Engineer
Requirements:
- 3+ years of experience in machine learning
- Strong Python and SQL skills; PyTorch or TensorFlow
- Experience with LangChain, RAG and Kubernetes
- Bachelor's degree in Computer Science
- AWS Certified Machine Learning preferred
Nice to have:
- Airflow, Spark
"""


@pytest.fixture(scope="session")
def resume_text():
    return (FIX / "resume_ml.txt").read_text()


@pytest.fixture(scope="session")
def encoder():
    from resume_intel.semantic import ConceptEncoder
    return ConceptEncoder()  # deterministic, no model download needed in CI


@pytest.fixture(scope="session")
def parsed(resume_text):
    from resume_intel.parser import parse_resume
    return parse_resume(resume_text, today=TODAY)


@pytest.fixture(scope="session")
def jd():
    from resume_intel.jobs import parse_job_description
    return parse_job_description(JD_ML)
