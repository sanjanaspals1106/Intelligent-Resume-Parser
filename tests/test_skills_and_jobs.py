from resume_intel import skills as sk
from resume_intel.jobs import parse_job_description


def names(t):
    return {c for c, _, _ in sk.find_skills(t)}


def test_aliases_normalise():
    assert {"kubernetes", "postgresql", "scikit-learn", "pytorch"} <= names("k8s, Postgres, sklearn, Torch")


def test_ambiguous_names_need_list_context():
    assert "c" not in names("I got a grade C in physics")
    assert "c" in names("Languages: Python, C, C++")
    assert "tensorflow" not in names("TF-IDF vectoriser")
    assert "go" not in names("We will Go to market")


def test_implied_and_related():
    assert sk.expand_implied({"pytorch"})["deep learning"] == "pytorch"
    assert "tensorflow" in sk.related_skills("pytorch")


def test_jd_levels_and_alternatives():
    jd = parse_job_description("""Engineer
Requirements:
- 4+ years of experience
- Python, PyTorch or TensorFlow
- Master's degree in Statistics
Nice to have:
- Docker
""")
    lv = {r.skill: r.level for r in jd.requirements}
    assert lv["python"] == "required" and lv["docker"] == "preferred"
    assert jd.min_years == 4 and jd.education_level == 4
    assert [r for r in jd.requirements if r.skill == "pytorch"][0].alternatives == ["tensorflow"]


def test_unstructured_jd_warns_and_treats_as_required():
    jd = parse_job_description("We need someone who knows Python and Docker to build things quickly.")
    assert set(jd.required_skills) == {"python", "docker"}
    assert any("no clear requirements" in w for w in jd.warnings)
