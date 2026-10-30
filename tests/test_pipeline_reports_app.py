import json
import re

from evaluation import synthetic as syn
from resume_intel.jobs import parse_job_description
from resume_intel.pipeline import analyze_batch, analyze_file, rank
from resume_intel.reports import candidate_html, candidate_markdown, ranking_csv, ranking_html, ranking_json
from tests.conftest import JD_ML, TODAY


def _batch(encoder):
    ps, _ = syn.build_corpus(9, per_family=[("strong", 1), ("weak", 1)])
    files = [syn.render(p) for p in ps if p.family == "ml_engineer"]
    files += [("broken.pdf", b"%PDF-1.4 nonsense"), ("empty.docx", b"")]
    return analyze_batch(files, JD_ML, encoder, TODAY), ps


def test_batch_reports_errors_and_ranks(encoder):
    res, _ = _batch(encoder)
    rows = rank(res)
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    assert len(ok) == 2 and len(bad) == 2 and all(r["error"] for r in bad)
    assert [r["rank"] for r in ok] == [1, 2] and ok[0]["score"] >= ok[1]["score"]
    assert rows[-1]["status"] == "error"   # failures listed last


def test_ranking_is_reproducible(encoder):
    a = rank(_batch(encoder)[0])
    b = rank(_batch(encoder)[0])
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]
    assert strip(a) == strip(b)


def test_reports_contain_disclaimer_and_escape_html(encoder):
    res, _ = _batch(encoder)
    a = [x for x in res if x.ok][0]
    a.resume.name = "<script>alert(1)</script>"
    h = candidate_html(a, "Role <b>")
    assert "<script>alert" not in h and "not a probability of being hired" in h
    assert "Recommendations" in candidate_markdown(a)
    rows = rank(res)
    assert "rank" in ranking_csv(rows).splitlines()[0]
    assert json.loads(ranking_json(rows, res, "x"))["ranking"]
    assert "Candidate 1" in ranking_html(rows, "x", anonymize=True)


def test_recommendation_templates_do_not_invent_numbers(encoder):
    res, _ = _batch(encoder)
    for a in res:
        if a.ok:
            for r in a.assessment.recommendations:
                if r["example"]:
                    assert not re.search(r"\d", r["example"]), r["example"]
                assert r["kind"] in {"improve_existing", "add_if_true", "develop", "restructure"}


def test_single_analysis_has_all_sections(parsed, encoder):
    ps, _ = syn.build_corpus(9, per_family=[("strong", 1)])
    fn, data = syn.render(ps[0])
    a = analyze_file(data, fn, parse_job_description(JD_ML), encoder, TODAY)
    d = a.to_dict()
    assert a.ok and d["assessment"]["checks"] and d["assessment"]["disclaimer"] and d["match"]["dimensions"]


def test_streamlit_app_smoke():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) >= 4


def test_real_data_ranking_mode(tmp_path, capsys):
    import json

    from evaluation.run import run_real
    ps, _ = syn.build_corpus(11, per_family=[("strong", 1), ("weak", 1)])
    rows = []
    for p in [x for x in ps if x.family == "ml_engineer"]:
        fn, data = syn.render(p)
        (tmp_path / fn).write_bytes(data)
        rows.append({"file": fn, "relevance": 3 if p.kind == "strong" else 0})
    spec = tmp_path / "labels.json"
    spec.write_text(json.dumps({"job_description": JD_ML, "resumes": rows}))
    run_real(str(spec))
    out = json.loads(capsys.readouterr().out)
    assert out["n_jobs"] == 1 and "baseline" in out and "hybrid" in out
