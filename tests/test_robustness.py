"""Fuzz-style checks: hostile, odd or huge text must never crash the pipeline."""
import random
import string
import time

from resume_intel.jobs import parse_job_description
from resume_intel.pipeline import analyze_text
from tests.conftest import JD_ML, TODAY

ALPHABET = string.printable + "éüñ日本語العربية🚀•–—“” ​\x00\x07"


def _jd():
    return parse_job_description(JD_ML)


def test_random_text_never_raises(encoder):
    rng = random.Random(1)
    jd = _jd()
    for _ in range(150):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(30, 1500)))
        a = analyze_text(text, jd, "fuzz", encoder, today=TODAY)
        assert a.status in ("ok", "error")
        if a.ok:
            assert 0 <= a.match.score <= 100


def test_structured_garbage_never_raises(encoder):
    jd = _jd()
    pieces = ["EXPERIENCE", "EDUCATION", "SKILLS:", "• ", "2020 - 2019", "Jan 2030 – Present", "B.Tech", "|||", "—", "Python,", ", , ,", "(((", "https://", "a@b", "+1", "Ph.D.", "0000-0000", "\n\n\n", "Engineer at at at", "C, R, Go"]
    rng = random.Random(2)
    for _ in range(200):
        text = " ".join(rng.choice(pieces) + rng.choice(["", "\n", " "]) for _ in range(rng.randint(5, 80)))
        text = "Some Name\n" + text + "\n" + "x" * 40
        a = analyze_text(text, jd, "g", encoder, today=TODAY)
        assert a.status in ("ok", "error")


def test_edge_job_descriptions(encoder):
    text = "Jane Doe\njane@example.com\n\nSKILLS\nPython, SQL\n\nEXPERIENCE\nAnalyst | Foo Ltd | 2020 - 2023\n- Wrote SQL reports for finance.\n"
    for jd_text in ("", "   ", "x", "🚀🚀🚀", "Python " * 5000, "Requirements:\n", "Nice to have:\n- Python"):
        a = analyze_text(text, parse_job_description(jd_text), "r", encoder, today=TODAY)
        assert a.status == "ok" and 0 <= a.match.score <= 100


def test_huge_resume_is_bounded_in_time(encoder):
    line = "• Built Python services with Docker and SQL for analytics workloads.\n"
    text = "Jane Doe\njane@example.com\nEXPERIENCE\nEngineer | Foo | 2015 - 2024\n" + line * 4000
    t0 = time.perf_counter()
    a = analyze_text(text, _jd(), "huge", encoder, today=TODAY)
    assert a.ok and time.perf_counter() - t0 < 60


def test_future_and_reversed_dates_do_not_create_negative_experience(encoder):
    text = "Jane Doe\njane@example.com\nEXPERIENCE\nEngineer | Foo | 2025 - 2021\n- Built things.\nAnalyst | Bar | Jan 2090 - Present\n- Did stuff.\n"
    a = analyze_text(text, _jd(), "d", encoder, today=TODAY)
    assert a.ok
    months = a.resume.total_experience_months
    assert months is None or months >= 0
