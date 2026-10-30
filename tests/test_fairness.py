"""Fairness / invariance audit: attributes that are not job-relevant must not change the score."""
import itertools
import re

import pytest

from evaluation.synthetic import FIRST, LAST
from resume_intel.matching import match
from resume_intel.parser import parse_resume
from tests.conftest import TODAY


def score(text, jd, enc):
    return match(parse_resume(text, today=TODAY), jd, enc).score


@pytest.mark.parametrize("first,last", list(itertools.islice(itertools.product(FIRST[:6], LAST[:6]), 0, 36, 3)))
def test_names_from_many_cultures_do_not_change_score(resume_text, jd, encoder, first, last):
    base = score(resume_text, jd, encoder)
    assert score(resume_text.replace("Priya Sharma", f"{first} {last}"), jd, encoder) == base


def test_institution_prestige_does_not_change_score(resume_text, jd, encoder):
    base = score(resume_text, jd, encoder)
    for inst in ("Podunk Community College", "Massachusetts Institute of Technology", "Unknown Technical Academy"):
        assert score(resume_text.replace("Indian Institute of Technology Madras", inst), jd, encoder) == base


def test_employment_gap_does_not_change_score(resume_text, jd, encoder):
    # same two jobs and durations, but with an 18-month gap inserted between them
    gapped = resume_text.replace("Jun 2021 - Dec 2021", "Jan 2020 - Jul 2020")
    assert gapped != resume_text
    assert score(gapped, jd, encoder) == score(resume_text, jd, encoder)


def test_interests_and_references_are_ignored(resume_text, jd, encoder):
    extra = resume_text + "\nINTERESTS\nGardening, chess, church volunteering\n\nREFERENCES\nAvailable on request\n"
    assert score(extra, jd, encoder) == score(resume_text, jd, encoder)


def test_missing_phone_and_links_do_not_change_score(resume_text, jd, encoder):
    stripped = re.sub(r" \| \+91 98765 43210 \| linkedin\.com/\S+ \| github\.com/\S+", "", resume_text)
    assert "linkedin" not in stripped
    assert score(stripped, jd, encoder) == score(resume_text, jd, encoder)


def test_education_dates_and_grades_do_not_change_score(resume_text, jd, encoder):
    assert score(resume_text.replace("2017 - 2021", "1985 - 1989").replace("8.7/10", "5.1/10"), jd, encoder) == score(resume_text, jd, encoder)


def test_wording_does_change_the_score_by_design(resume_text, jd, encoder):
    """Documented limitation: relevant wording is evidence, so removing it lowers the score."""
    no_k8s = resume_text.replace("k8s, ", "")
    assert score(no_k8s, jd, encoder) < score(resume_text, jd, encoder)
