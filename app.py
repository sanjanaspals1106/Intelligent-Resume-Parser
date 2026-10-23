import re
import fitz
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Intelligent Resume Parser", page_icon="📄", layout="wide")

st.title("Intelligent Resume Parser & Job Matching")
st.write("Upload a resume and compare it with a job description.")

SKILLS = [
    "Python", "Java", "C++", "SQL", "JavaScript", "TypeScript",
    "Machine Learning", "Deep Learning", "NLP", "Generative AI",
    "LLM", "Prompt Engineering", "Computer Vision", "PyTorch",
    "TensorFlow", "Scikit-learn", "Pandas", "NumPy", "Data Analysis",
    "Statistics", "FastAPI", "Flask", "REST API", "HTML", "CSS",
    "React", "Node.js", "Git", "GitHub", "PostgreSQL", "MySQL",
    "MongoDB", "Docker", "AWS", "Azure", "Data Structures",
    "Algorithms", "Excel", "Power BI", "Tableau", "OpenCV", "YOLO",
    "Hugging Face", "Transformers", "FAISS", "Streamlit"
]

def extract_text(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document).strip()

def detect_skills(text):
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(skill)
    return sorted(set(found))

def detect_email(text):
    match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else "Not detected"

def detect_phone(text):
    for value in re.findall(r"\+?\d[\d ()-]{8,}\d", text):
        digits = re.sub(r"\D", "", value)
        if 10 <= len(digits) <= 15:
            return value.strip()
    return "Not detected"

uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
job_description = st.text_area(
    "Job description",
    placeholder="Paste the job description here...",
    height=200
)

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a PDF resume.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        try:
            resume_text = extract_text(uploaded_file.getvalue())

            if len(resume_text) < 30:
                st.error(
                    "Not enough text was extracted. This may be a scanned PDF; "
                    "OCR is not included."
                )
                st.stop()

            resume_skills = detect_skills(resume_text)
            job_skills = detect_skills(job_description)

            matched_skills = sorted(set(resume_skills) & set(job_skills))
            missing_skills = sorted(set(job_skills) - set(resume_skills))

            skill_coverage = (
                len(matched_skills) / len(set(job_skills)) * 100
                if job_skills else 0.0
            )

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )
            matrix = vectorizer.fit_transform(
                [resume_text.lower(), job_description.lower()]
            )
            similarity = float(
                cosine_similarity(matrix[0:1], matrix[1:2])[0, 0]
            ) * 100

            score = (
                0.6 * skill_coverage + 0.4 * similarity
                if job_skills else similarity
            )

            st.subheader("Match Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Baseline Match Score", f"{score:.1f}%")
            col2.metric("Skill Coverage", f"{skill_coverage:.1f}%")
            col3.metric("Text Similarity", f"{similarity:.1f}%")

            st.caption(
                "The baseline score combines skill coverage (60%) and TF-IDF "
                "similarity (40%) when job skills are detected. These are "
                "heuristic relevance scores, not hiring probabilities."
            )

            st.subheader("Contact Details")
            st.write("**Email:**", detect_email(resume_text))
            st.write("**Phone:**", detect_phone(resume_text))

            st.subheader("Skills Detected in Resume")
            st.write(", ".join(resume_skills) if resume_skills else "No listed skills detected.")

            st.subheader("Matched Skills")
            st.write(", ".join(matched_skills) if matched_skills else "No direct skill matches found.")

            st.subheader("Potential Skill Gaps")
            st.write(", ".join(missing_skills) if missing_skills else "No gaps detected using the current skill dictionary.")

            with st.expander("View Extracted Resume Text"):
                st.text(resume_text)

            st.download_button(
                "Download Analysis as TXT",
                data=(
                    f"Match score: {score:.1f}%\n"
                    f"Skill coverage: {skill_coverage:.1f}%\n"
                    f"Text similarity: {similarity:.1f}%\n"
                    f"Detected skills: {', '.join(resume_skills)}\n"
                    f"Matched skills: {', '.join(matched_skills)}\n"
                    f"Potential skill gaps: {', '.join(missing_skills)}\n"
                ),
                file_name="resume_analysis.txt",
                mime="text/plain"
            )

            st.info(
                "This baseline app uses a finite skill dictionary and literal "
                "matching. Review results manually; do not use this score as "
                "the sole basis for hiring decisions."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")
