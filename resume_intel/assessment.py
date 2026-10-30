"""ATS-style, job-specific assessment built on top of a ``MatchResult``.

"ATS-style" means generic applicant-tracking-system practices (parseable text, standard sections, dates,
keyword evidence). It is not a simulation of, or a compatibility claim about, any commercial ATS product.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .jobs import JobDescription
from .matching import MatchResult
from .parser import ACTION_VERBS, RESULT_RE, ParsedResume
from .recommendations import Recommendation, recommend

DISCLAIMER = ("This score estimates how relevant the resume content is to this job description. It is not a probability of being hired, "
              "not a guarantee of passing any applicant tracking system, and does not simulate any specific commercial ATS. Use it as one input to human review.")

WEAK_OPENERS = re.compile(r"^(responsible for|worked on|helped|involved in|duties included|assisted with|participated in|tasked with|was part of)\b", re.I)


@dataclass
class Check:
    key: str
    label: str
    status: str  # pass | warn | fail
    detail: str


@dataclass
class Assessment:
    score: float
    band: str
    confidence: str
    dimensions: list
    completeness_score: float
    readability_score: float
    checks: list
    strengths: list
    gaps: list
    uncertainties: list
    recommendations: list
    disclaimer: str = DISCLAIMER
    stats: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def bullet_lines(resume: ParsedResume) -> list[str]:
    out = []
    for e in resume.experience:
        out += [h for h in e.highlights if h.strip()]
    for p in resume.projects:
        out += [d for d in p.description if d.strip()]
    return out


def has_action_verb(line: str) -> bool:
    first = re.sub(r"[^a-z\-]", "", line.split()[0].lower()) if line.split() else ""
    return first in ACTION_VERBS or (first.endswith("ed") and len(first) > 4 and not WEAK_OPENERS.match(line))


def _checks(resume: ParsedResume) -> tuple[list[Check], dict]:
    cs: list[Check] = []

    def add(key, label, ok, detail, warn=False):
        cs.append(Check(key, label, "pass" if ok else ("warn" if warn else "fail"), detail))

    add("email", "E-mail address detected", bool(resume.email), resume.email and "Found." or "No e-mail address was detected.")
    add("phone", "Phone number detected", bool(resume.phone), "Found." if resume.phone else "No phone number was detected.", warn=True)
    add("links", "Professional links (LinkedIn / GitHub / portfolio)", bool(resume.links), ", ".join(k for k in resume.links) or "None detected; optional but helpful for technical roles.", warn=True)
    add("sections", "Standard section headings", {"experience", "education", "skills"} <= set(resume.sections_found),
        f"Detected: {', '.join(resume.sections_found) or 'none'}.", warn=bool(resume.sections_found))
    add("summary", "Professional summary / profile", bool(resume.summary), "Found." if resume.summary else "No summary section; optional.", warn=True)
    add("text_layer", "Text is machine-readable", resume.extraction_method in ("pdf-text", "docx", "txt", "text"),
        {"text": "Plain text.", "pdf-text": "Native text layer.", "docx": "DOCX text.", "txt": "Plain text."}.get(resume.extraction_method, "OCR was required; automated parsers may read this file poorly."), warn=True)
    dated = [e for e in resume.experience if e.dates.start or e.dates.raw]
    add("dates", "Experience entries have dates", bool(resume.experience) and len(dated) == len(resume.experience),
        f"{len(dated)}/{len(resume.experience)} entries dated." if resume.experience else "No experience entries parsed.", warn=bool(dated))
    add("titles", "Experience entries have title and organisation", bool(resume.experience) and all(e.title and e.organization for e in resume.experience),
        f"{sum(bool(e.title and e.organization) for e in resume.experience)}/{len(resume.experience)} complete." if resume.experience else "No experience entries parsed.", warn=bool(resume.experience))
    lines = bullet_lines(resume)
    n = len(lines)
    verbs = sum(has_action_verb(l) for l in lines)
    nums = sum(bool(RESULT_RE.search(l)) for l in lines)
    words = [len(l.split()) for l in lines]
    avg = sum(words) / n if n else 0
    add("action_verbs", "Bullets start with action verbs", n > 0 and verbs / n >= 0.6, f"{verbs}/{n} bullets." if n else "No bullets found.", warn=n > 0 and verbs / n >= 0.35)
    add("quantified", "Some achievements are quantified", n > 0 and nums / n >= 0.2, f"{nums}/{n} bullets contain a figure or recognised outcome." if n else "No bullets found.", warn=n > 0 and nums > 0)
    add("bullet_length", "Bullet length is scannable (8-35 words)", n > 0 and 8 <= avg <= 35, f"Average {avg:.0f} words." if n else "No bullets found.", warn=n > 0)
    wc = len(resume.text.split())
    add("length", "Length is reasonable (250-1100 words)", 250 <= wc <= 1100, f"{wc} words.", warn=150 <= wc <= 1600)
    listed = [s for s in resume.skills if s.listed]
    ev = [s for s in listed if set(s.sections) & {"experience", "projects"}]
    add("skills_evidenced", "Listed skills are demonstrated in experience/projects", bool(listed) and len(ev) / len(listed) >= 0.5,
        f"{len(ev)}/{len(listed)} listed skills appear in experience or projects." if listed else "No skills list found.", warn=bool(listed))
    odd = len(re.findall(r"[^\w\s.,;:()@+#/&%\-–—'\"•|\[\]!?$=*]", resume.text)) / max(1, len(resume.text))
    add("noise", "Little garbled / special-character noise", odd < 0.01, f"{odd * 100:.2f}% unusual characters.", warn=odd < 0.03)
    stats = {"bullets": n, "action_verb_share": round(verbs / n, 2) if n else None, "quantified_share": round(nums / n, 2) if n else None, "avg_bullet_words": round(avg, 1), "word_count": wc}
    return cs, stats


def _score(cs, keys):
    sel = [c for c in cs if c.key in keys]
    return round(100 * sum({"pass": 1.0, "warn": 0.5, "fail": 0.0}[c.status] for c in sel) / len(sel), 1) if sel else 0.0


def assess(resume: ParsedResume, jd: JobDescription, m: MatchResult) -> Assessment:
    cs, stats = _checks(resume)
    completeness = _score(cs, {"email", "phone", "links", "sections", "summary", "dates", "titles", "skills_evidenced"})
    readability = _score(cs, {"text_layer", "action_verbs", "bullet_length", "length", "noise", "quantified"})

    strengths, gaps, unc = [], [], []
    matched_req = [s for s in m.skill_matches if s.level == "required" and s.status in ("matched", "implied") and not s.listed_only]
    if matched_req:
        strengths.append({"text": f"{len(matched_req)} of {sum(s.level == 'required' for s in m.skill_matches)} required skills are evidenced: " + ", ".join(s.skill for s in matched_req[:8]) + ("…" if len(matched_req) > 8 else ""),
                          "evidence": next((e["text"] for s in matched_req for e in s.evidence), None)})
    for d in m.dimensions:
        if d.applicable and d.score is not None and d.score >= 0.75 and d.key in ("experience", "projects", "education", "certifications", "semantic"):
            strengths.append({"text": f"{d.label} is strong ({d.score:.0%}).", "evidence": d.detail})
    if m.experience_evidence and m.experience_evidence[0]["relevance"] >= 0.4:
        e = m.experience_evidence[0]
        strengths.append({"text": f"Most relevant experience: {e['title'] or 'role'}" + (f" at {e['organization']}" if e["organization"] else ""), "evidence": (e["lines"] or [None])[0]})
    if m.project_evidence and m.project_evidence[0]["relevance"] >= 0.4:
        strengths.append({"text": f"Most relevant project: {m.project_evidence[0]['title']}", "evidence": (m.project_evidence[0]["lines"] or [None])[0]})

    for s in m.skill_matches:
        if s.status == "missing" and not s.uncertain:
            gaps.append({"text": f"{s.level.capitalize()} skill not evidenced: {s.skill}" + (f" (or {', '.join(s.alternatives)})" if s.alternatives else ""), "level": s.level})
        elif s.status == "missing":
            unc.append({"text": f"{s.skill} ({s.level}) was not found, but the resume was only partly parsed, so it may be present and not extracted."})
        elif s.status == "related":
            gaps.append({"text": f"{s.skill} ({s.level}) not evidenced directly; related skill '{s.via}' is present.", "level": s.level})
        elif s.listed_only:
            unc.append({"text": f"{s.skill} is listed but not demonstrated in any experience or project, so evidence is thin."})
    for d in m.dimensions:
        if d.applicable and d.score is not None and d.score < 0.35 and d.key in ("experience", "projects", "semantic"):
            gaps.append({"text": f"Low {d.label.lower()} ({d.score:.0%}).", "level": "info"})
    if m.education.get("applicable") and (m.education.get("score") or 0) < 0.6:
        (unc if m.education.get("uncertain") else gaps).append({"text": "Education: " + m.education["detail"], "level": "required"})
    for c in m.certifications:
        if c["status"] != "matched":
            gaps.append({"text": f"Certification ({c['level']}): {c['requirement'][:90]} -> {c['status']}", "level": c["level"]})
    if jd.min_years and resume.total_experience_months is None:
        unc.append({"text": f"The job asks for {jd.min_years}+ years of experience but dates could not be extracted to verify this."})
    for w in resume.warnings:
        unc.append({"text": w})
    for w in m.warnings:
        if w not in [u["text"] for u in unc]:
            unc.append({"text": w})

    recs = recommend(resume, jd, m, stats)
    return Assessment(score=m.score, band=m.band, confidence=m.confidence, dimensions=[asdict(d) for d in m.dimensions], completeness_score=completeness,
                      readability_score=readability, checks=[asdict(c) for c in cs], strengths=strengths, gaps=gaps, uncertainties=unc,
                      recommendations=[asdict(r) for r in recs], stats=stats)
