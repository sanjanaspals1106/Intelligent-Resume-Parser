import copy

import pytest

from evaluation import synthetic as syn
from resume_intel.baseline import baseline_match
from resume_intel.jobs import parse_job_description
from resume_intel.matching import DEFAULT_WEIGHTS, match
from resume_intel.parser import parse_resume
from tests.conftest import TODAY


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1) < 1e-9


def test_baseline_preserved():
    b = baseline_match("I know Python and SQL and Docker.", "Need Python, SQL and AWS.")
    assert b["matched"] == ["python", "sql"] and b["missing"] == ["aws"]
    assert round(b["skill_coverage"], 1) == 66.7


def test_score_breakdown_is_consistent(parsed, jd, encoder):
    m = match(parsed, jd, encoder)
    assert abs(sum(d.contribution for d in m.dimensions) - m.score) < 0.05
    assert abs(sum(d.weight for d in m.dimensions if d.applicable) - 1) < 1e-3
    assert 0 <= m.score <= 100 and m.band


def test_requirement_statuses(parsed, jd, encoder):
    st = {s.skill: s for s in match(parsed, jd, encoder).skill_matches}
    assert st["python"].status == "matched" and st["langchain"].status == "matched"
    assert st["pytorch"].status == "matched"           # alternative group PyTorch or TensorFlow
    assert st["kubernetes"].listed_only and st["kubernetes"].credit < 1   # only in skills list
    assert "tensorflow" not in st                       # alternatives collapse into one requirement


def test_missing_skill_is_not_confused_with_unparsed(jd, encoder):
    r = parse_resume("Jane Doe\n\nSKILLS\nPython\n", today=TODAY)  # weakly parsed
    m = match(r, jd, encoder)
    miss = [s for s in m.skill_matches if s.status == "missing"]
    assert miss and all(s.uncertain for s in miss)
    good = parse_resume(open("tests/fixtures/resume_ml.txt").read(), today=TODAY)
    kafka = [s for s in match(good, parse_job_description("Requirements:\n- Kafka experience\n"), encoder).skill_matches][0]
    assert kafka.status == "missing" and not kafka.uncertain


def test_related_skill_gets_partial_credit(parsed, encoder):
    jd2 = parse_job_description("Requirements:\n- Experience with MongoDB or Cassandra\n- Azure\n")
    st = {s.skill: s for s in match(parsed, jd2, encoder).skill_matches}
    assert st["azure"].status == "related" and st["azure"].via == "aws" and st["azure"].credit == 0.4


def test_ranking_orders_strong_above_weak_and_stuffed(encoder):
    mix = [("strong", 1), ("weak", 1), ("stuffed", 1)]
    ps, jobs = syn.build_corpus(5, per_family=mix)
    job = [j for j in jobs if j["id"] == "J3"][0]
    jdp = parse_job_description(job["text"])
    sc = {}
    for p in ps:
        if p.family == "backend_dev":
            sc[p.kind] = match(parse_resume("\n".join(t for _, t in p.structure), today=TODAY), jdp, encoder).score
    assert sc["strong"] > sc["weak"] and sc["strong"] > sc["stuffed"]


def test_name_and_contact_do_not_affect_score(resume_text, jd, encoder):
    a = parse_resume(resume_text, today=TODAY)
    swapped = resume_text.replace("Priya Sharma", "John Smith").replace("priya.sharma@example.com", "j.smith@corp.org").replace("+91 98765 43210", "+1 202 555 0143")
    b = parse_resume(swapped, today=TODAY)
    assert a.name != b.name
    assert match(a, jd, encoder).score == match(b, jd, encoder).score


def test_education_dates_do_not_affect_score(resume_text, jd, encoder):
    a = parse_resume(resume_text, today=TODAY)
    b = parse_resume(resume_text.replace("2017 - 2021", "1990 - 1994"), today=TODAY)
    assert match(a, jd, encoder).score == match(b, jd, encoder).score


def test_scoring_view_excludes_pii(parsed):
    blob = str(parsed.scoring_view())
    assert parsed.email not in blob and parsed.name not in blob and parsed.phone not in blob


def test_deterministic(parsed, jd, encoder):
    assert match(parsed, jd, encoder).score == match(parsed, jd, encoder).score
