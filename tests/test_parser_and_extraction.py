import io

import pytest

from evaluation import synthetic as syn
from resume_intel.extraction import ExtractionError, extract_text, ocr_available
from resume_intel.parser import parse_resume
from tests.conftest import TODAY


def test_contact_and_links(parsed):
    assert parsed.name == "Priya Sharma" and parsed.name_confidence == "high"
    assert parsed.email == "priya.sharma@example.com"
    assert parsed.phone.replace(" ", "") == "+919876543210"
    assert "linkedin" in parsed.links and "github" in parsed.links


def test_experience_dates_titles(parsed):
    assert [e.title for e in parsed.experience] == ["Machine Learning Engineer", "Data Science Intern"]
    assert parsed.experience[0].organization == "Acme Analytics Pvt Ltd"
    assert parsed.experience[0].dates.months == 37 and parsed.experience[0].dates.end == "present"
    assert parsed.experience[1].is_internship
    assert parsed.total_experience_months == 37 and parsed.internship_months == 7


def test_education_projects_certs(parsed):
    e = parsed.education[0]
    assert e.level == 3 and "Technology" in e.institution and e.field_of_study.startswith("Computer Science")
    assert [p.name for p in parsed.projects] == ["Resume Parser", "Movie Recommender"]
    assert "flask" in parsed.projects[1].technologies
    assert parsed.certifications[0].issuer == "Amazon Web Services"


def test_skill_records_distinguish_listed_from_evidenced(parsed):
    rec = {s.name: s for s in parsed.skills}
    assert rec["pytorch"].listed and "experience" in rec["pytorch"].sections and rec["pytorch"].evidence
    assert rec["tableau"].listed and not rec["tableau"].evidence  # only in the skills list


def test_missing_sections_are_reported_not_invented():
    r = parse_resume("Jane Doe\njane@example.com\n\nSKILLS\nPython, SQL\n", today=TODAY)
    assert r.experience == [] and r.education == [] and r.total_experience_months is None
    assert "experience" in r.sections_missing and any("experience" in w for w in r.warnings)


def test_no_headings_gives_warning():
    r = parse_resume("Just a blob of text about Python and Docker usage in many projects over many years.", today=TODAY)
    assert any("No recognised section headings" in w for w in r.warnings)
    assert r.parse_quality < 0.5


def test_pdf_and_docx_roundtrip():
    ps, _ = syn.build_corpus(3, per_family=[("strong", 1)])
    for p in ps[:4]:
        fn, data = syn.render(p)
        ex = extract_text(data, fn)
        r = parse_resume(ex.text, ex.links, ex.method, TODAY)
        assert r.email == p.email and r.name == p.name
        assert len(r.experience) == len(p.jobs)


@pytest.mark.skipif(not ocr_available(), reason="Tesseract not installed")
def test_scanned_pdf_uses_ocr():
    ps, _ = syn.build_corpus(3, per_family=[("strong", 1)])
    p = [x for x in ps if x.layout == "classic"][0]
    ex = extract_text(syn.render(p, scanned=True)[1], "scan.pdf")
    assert ex.ocr_used and ex.method in ("pdf-ocr", "pdf-text+ocr")
    assert p.email.split("@")[0][:6].lower() in ex.text.lower()


@pytest.mark.parametrize("data,name", [(b"", "a.pdf"), (b"%PDF-1.4 garbage", "a.pdf"), (b"PK\x03\x04junk", "a.docx"), (b"hello" * 30, "a.rtf"), (b"\xd0\xcf\x11\xe0" + b"0" * 50, "a.doc")])
def test_malformed_inputs_raise_extraction_error(data, name):
    with pytest.raises(ExtractionError):
        extract_text(data, name)
