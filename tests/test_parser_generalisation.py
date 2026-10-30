"""Regression tests for parser behaviours added after the round-1 held-out failure analysis."""
from datetime import date

from evaluation import heldout as ho
from resume_intel.extraction import extract_text
from resume_intel.parser import parse_resume

T = date(2025, 6, 1)


def test_role_header_without_dates_starts_new_entry():
    r = parse_resume("Jo Bloggs\njo@example.com\n\nExperience\nData Engineer, Acme Ltd\n• Built pipelines.\n• Fixed jobs.\nAnalyst, Beta Corp\n• Wrote reports.\n", today=T)
    assert [(e.title, e.organization) for e in r.experience] == [("Data Engineer", "Acme Ltd"), ("Analyst", "Beta Corp")]


def test_headingless_resume_infers_experience_with_low_confidence():
    r = parse_resume("Jo Bloggs\njo@example.com\n\nAnalyst | Beta Corp | 2021 - 2023\nWrote SQL reports.\nSkills: SQL, Python\n", today=T)
    assert len(r.experience) == 1 and r.experience[0].confidence == "low" and r.experience[0].title == "Analyst"
    assert any("inferred" in w for w in r.warnings)


def test_free_text_line_is_not_an_employer():
    r = parse_resume("Jo\njo@example.com\n\nExperience\nNo professional experience yet; see projects.\n\nSkills\nPython, SQL\n", today=T)
    assert r.experience == []


def test_short_description_sentences_are_not_project_titles():
    r = parse_resume("Jo\njo@example.com\n\nProjects\nQuill Hub\nUsed Python as a hobby.\nUsed Docker too.\n\nCobalt Tracker\nBuilt with SQL.\n", today=T)
    assert [p.name for p in r.projects] == ["Quill Hub", "Cobalt Tracker"]


def test_selected_work_is_a_projects_heading_and_bootcamp_counts_as_education():
    r = parse_resume("Jo\njo@example.com\n\nSELECTED WORK\nAtlas\nBuilt things with Python.\n\nEducation\nNextstep Coding Bootcamp - Full-stack certificate (2023)\n", today=T)
    assert "projects" in r.sections_found and r.projects[0].name == "Atlas"
    assert r.education and r.education[0].level == 2


def test_two_column_pdf_is_read_column_by_column():
    cands, _ = ho.build(3, per_arch=[("exact", 1)], layouts=("L5",))
    c = cands[0]
    fn, data = ho.render_cand(c)
    text = extract_text(data, fn).text
    r = parse_resume(text, today=T)
    assert [e.title for e in r.experience] == [j[0] for j in c.jobs]
    assert r.education and r.education[0].institution == c.education[0][2]


def test_unicode_dashes_in_dates_are_parsed():
    r = parse_resume("Jo\njo@example.com\n\nExperience\nEngineer | Acme | Jan 2020 – Mar 2022\n• Did work.\nAnalyst | Beta | 2018 — 2019\n• More work.\n", today=T)
    assert [e.dates.months for e in r.experience] == [27, 24]


def test_heldout_generator_is_deterministic_across_hash_seeds():
    import hashlib
    import os
    import subprocess
    import sys
    code = ("import hashlib;from evaluation import heldout as h;cs,_=h.build(5,layouts=('L1','L5'));"
            "print(hashlib.sha256('|'.join(str(c.structure) for c in cs).encode()).hexdigest()[:12])")
    outs = {subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": hs}).stdout.strip() for hs in ("1", "2", "3")}
    assert len(outs) == 1 and "" not in outs
