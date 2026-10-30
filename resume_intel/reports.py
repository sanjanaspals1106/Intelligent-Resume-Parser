"""Downloadable reports: JSON, CSV, Markdown and self-contained HTML (printable to PDF)."""
from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime

from .assessment import DISCLAIMER
from .pipeline import Analysis, anonymized_labels

_E = html.escape


def ranking_csv(rows: list[dict]) -> str:
    cols = ["rank", "candidate", "file", "score", "band", "required_coverage", "preferred_coverage", "matched_required", "missing_required", "possibly_missed",
            "experience_years", "baseline_score", "confidence", "completeness", "near_tie_with_next", "status", "error"]
    dims = sorted({k for r in rows for k in r if k.startswith("dim_")})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols + dims, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        r2 = dict(r)
        for k in ("matched_required", "missing_required", "possibly_missed"):
            if isinstance(r2.get(k), list):
                r2[k] = "; ".join(r2[k])
        w.writerow(r2)
    return buf.getvalue()


def ranking_json(rows: list[dict], analyses: list[Analysis], job_title: str | None, anonymize: bool = False) -> str:
    labels = anonymized_labels(analyses) if anonymize else {}
    return json.dumps({"generated": datetime.now().isoformat(timespec="seconds"), "job_title": job_title, "disclaimer": DISCLAIMER, "anonymized": anonymize,
                       "ranking": rows, "candidates": [a.to_dict(anonymize_as=labels.get(a.filename)) for a in analyses]}, indent=2, default=str)


def _list(items):
    return "<ul>" + "".join(f"<li>{_E(str(i))}</li>" for i in items) + "</ul>" if items else "<p class='muted'>None</p>"


_CSS = """body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d2433;line-height:1.5}
h1{font-size:1.6rem}h2{border-bottom:1px solid #d8dde6;padding-bottom:.2rem;margin-top:2rem}.muted{color:#667085}
table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid #d8dde6;padding:.35rem .5rem;text-align:left;vertical-align:top}th{background:#f3f5f9}
.score{font-size:2.2rem;font-weight:700}.pill{display:inline-block;padding:.1rem .6rem;border-radius:99px;background:#eef2ff;font-size:.8rem}
.bar{background:#e5e9f2;border-radius:4px;height:10px;width:140px;display:inline-block}.bar>i{display:block;height:10px;background:#3b5bdb;border-radius:4px}
.note{background:#fff8e1;border-left:4px solid #f5b301;padding:.6rem .9rem;margin:1rem 0;font-size:.9rem}@media print{body{margin:0}}"""


def _page(title, body):
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_E(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"


def candidate_html(a: Analysis, job_title: str | None = None, anonymize: bool = False) -> str:
    if not a.ok:
        return _page("Analysis failed", f"<h1>{_E(a.filename)}</h1><p>Analysis failed: {_E(a.error or '')}</p>")
    m, r, s = a.match, a.resume, a.assessment
    who = "Candidate" if anonymize else (r.name or a.filename)
    b = [f"<h1>Resume assessment: {_E(who)}</h1><p class='muted'>Target role: {_E(job_title or 'n/a')} · generated {datetime.now():%Y-%m-%d %H:%M}</p>",
         f"<div class='score'>{m.score:.0f}/100 <span class='pill'>{_E(m.band)}</span></div><p>Parse confidence: <b>{_E(m.confidence)}</b> · Completeness {s.completeness_score:.0f}/100 · Readability {s.readability_score:.0f}/100 · Baseline (original method): {m.baseline['score']:.0f}</p>",
         f"<div class='note'>{_E(DISCLAIMER)}</div>", "<h2>How the score was built</h2><table><tr><th>Factor</th><th>Score</th><th>Weight</th><th>Points</th><th>Explanation</th></tr>"]
    for d in m.dimensions:
        if d.applicable:
            b.append(f"<tr><td>{_E(d.label)}</td><td><span class='bar'><i style='width:{100 * (d.score or 0):.0f}%'></i></span> {100 * (d.score or 0):.0f}%</td><td>{100 * d.weight:.0f}%</td><td>{d.contribution:.1f}</td><td>{_E(d.detail)}</td></tr>")
        else:
            b.append(f"<tr><td>{_E(d.label)}</td><td colspan=3 class='muted'>not applicable</td><td>{_E(d.detail)}</td></tr>")
    b.append("</table><h2>Skill requirements</h2><table><tr><th>Skill</th><th>Level</th><th>Result</th><th>Evidence / note</th></tr>")
    for k in m.skill_matches:
        ev = k.evidence[0]["text"] if k.evidence else ""
        b.append(f"<tr><td>{_E(k.skill)}</td><td>{_E(k.level)}</td><td>{_E(k.status)}{' (possibly not extracted)' if k.uncertain else ''}</td><td>{_E(ev)} <span class='muted'>{_E(k.note)}</span></td></tr>")
    b.append("</table>")
    for title, items in (("Strengths", [x["text"] + (f" — “{x['evidence']}”" if x.get("evidence") else "") for x in s.strengths]), ("Gaps (not evidenced)", [x["text"] for x in s.gaps]), ("Uncertainties (may be extraction limits)", [x["text"] for x in s.uncertainties])):
        b.append(f"<h2>{title}</h2>{_list(items)}")
    b.append("<h2>Relevant experience</h2>")
    b.append(_list([f"{e['title']} {('@ ' + e['organization']) if e['organization'] else ''} {e['dates']} — relevance {e['relevance']:.2f}; skills: {', '.join(e['matched_skills']) or '-'}" for e in m.experience_evidence]))
    b.append("<h2>Relevant projects</h2>" + _list([f"{e['title']} — relevance {e['relevance']:.2f}; skills: {', '.join(e['matched_skills']) or '-'}" for e in m.project_evidence]))
    b.append("<h2>Recommendations</h2><p class='muted'>Kinds: improve_existing (reword what is there) · add_if_true (only if genuinely true) · develop (a real gap) · restructure (parseability). Nothing here should be added to a resume unless it is accurate.</p>")
    for rec in s.recommendations:
        b.append(f"<p><b>[{_E(rec['priority'])}] {_E(rec['title'])}</b> <span class='pill'>{_E(rec['kind'])}</span><br>{_E(rec['detail'])}" + (f"<br><i>From your resume:</i> {_E(rec['evidence'])}" if rec.get("evidence") else "") + (f"<br><i>Template:</i> <code>{_E(rec['example'])}</code>" if rec.get("example") else "") + "</p>")
    return _page(f"Assessment - {who}", "".join(b))


def candidate_markdown(a: Analysis, job_title: str | None = None, anonymize: bool = False) -> str:
    if not a.ok:
        return f"# {a.filename}\n\nAnalysis failed: {a.error}\n"
    m, r, s = a.match, a.resume, a.assessment
    o = [f"# Resume assessment: {'Candidate' if anonymize else (r.name or a.filename)}", f"Target role: {job_title or 'n/a'}", "", f"**Compatibility: {m.score:.0f}/100 ({m.band})** · parse confidence {m.confidence}", "", f"> {DISCLAIMER}", "",
         "## Score breakdown", "| Factor | Score | Weight | Points |", "|---|---|---|---|"]
    for d in m.dimensions:
        o.append(f"| {d.label} | {'n/a' if d.score is None else f'{100 * d.score:.0f}%'} | {100 * d.weight:.0f}% | {d.contribution:.1f} |")
    o += ["", "## Skills"] + [f"- **{k.skill}** ({k.level}): {k.status}" + (" (possibly not extracted)" if k.uncertain else "") for k in m.skill_matches]
    o += ["", "## Strengths"] + [f"- {x['text']}" for x in s.strengths] + ["", "## Gaps"] + [f"- {x['text']}" for x in s.gaps] + ["", "## Uncertainties"] + [f"- {x['text']}" for x in s.uncertainties]
    o += ["", "## Recommendations"] + [f"- **[{x['priority']}] {x['title']}** (`{x['kind']}`): {x['detail']}" for x in s.recommendations]
    return "\n".join(o) + "\n"


def ranking_html(rows: list[dict], job_title: str | None, anonymize: bool = False) -> str:
    b = [f"<h1>Candidate ranking</h1><p class='muted'>Role: {_E(job_title or 'n/a')} · {datetime.now():%Y-%m-%d %H:%M}</p><div class='note'>{_E(DISCLAIMER)} Rankings are decision support for human reviewers; names, contact details and demographic information are never used in scoring.</div>",
         "<table><tr><th>#</th><th>Candidate</th><th>Score</th><th>Band</th><th>Req. coverage</th><th>Missing required</th><th>Confidence</th><th>Baseline</th></tr>"]
    for r in rows:
        if r.get("status") != "ok":
            b.append(f"<tr><td>–</td><td>{_E(r['file'])}</td><td colspan=6>Failed: {_E(r.get('error', ''))}</td></tr>")
            continue
        name = f"Candidate {r['rank']}" if anonymize else r["candidate"]
        miss = ", ".join(r["missing_required"]) + (" (may be extraction gaps: " + ", ".join(r["possibly_missed"]) + ")" if r["possibly_missed"] else "")
        b.append(f"<tr><td>{r['rank']}</td><td>{_E(name)}{' ≈ tie with next' if r['near_tie_with_next'] else ''}</td><td>{r['score']}</td><td>{_E(r['band'])}</td><td>{r['required_coverage']}%</td><td>{_E(miss)}</td><td>{_E(r['confidence'])}</td><td>{r['baseline_score']}</td></tr>")
    b.append("</table>")
    return _page("Candidate ranking", "".join(b))
