"""Rule-based, evidence-grounded resume recommendations (no external AI service).

Every recommendation is one of four kinds, so the reader knows whether it asks them to:
  * improve_existing -- rewrite/reorder something already on the resume
  * add_if_true      -- surface genuine information that may have been omitted (only if it is true)
  * develop          -- build a skill/qualification that is genuinely not evidenced
  * restructure      -- make the document easier to parse
Templates use [bracketed placeholders]; the system never invents skills, outcomes or numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import skills as sk
from .jobs import JobDescription
from .parser import RESULT_RE, ParsedResume

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Recommendation:
    kind: str
    priority: str
    title: str
    detail: str
    evidence: Optional[str] = None
    example: Optional[str] = None


def _weak_bullets(resume: ParsedResume):
    from .assessment import WEAK_OPENERS, bullet_lines, has_action_verb

    return [l for l in bullet_lines(resume) if WEAK_OPENERS.match(l) or not has_action_verb(l)]


def recommend(resume: ParsedResume, jd: JobDescription, m, stats: dict) -> list[Recommendation]:
    from .assessment import bullet_lines

    recs: list[Recommendation] = []
    add = lambda *a, **k: recs.append(Recommendation(*a, **k))

    # --- 1. requirements not evidenced ------------------------------------------------------
    for s in [x for x in m.skill_matches if x.level == "required"]:
        names = f"{s.skill}" + (f" (or {', '.join(s.alternatives)})" if s.alternatives else "")
        if s.status == "related":
            add("add_if_true", "high", f"Clarify your experience with {names}",
                f"The job requires {names}. Your resume shows related experience with '{s.via}' but never mentions {s.skill}. If you have genuinely used {s.skill} (work, coursework, a personal project), say where and what you did. If not, treat it as a skill to develop: your '{s.via}' background is a reasonable starting point.",
                evidence=(s.evidence[0]["text"] if s.evidence else None),
                example=f"[Action verb] [what you built/did] using {s.skill} to [purpose]. (Only if accurate.)")
        elif s.status == "missing" and s.uncertain:
            add("restructure", "medium", f"Make {names} easy to detect",
                f"{names} was not found, but the resume was only partly parsed. If you do have this skill, make sure it appears as plain text (not in an image or text box) in a Skills section and in the relevant experience or project bullet.")
        elif s.status == "missing":
            add("develop", "high", f"Required skill not evidenced: {names}",
                f"Nothing in the resume shows {names}, which the job lists as required. If you have used it, add it with context (add_if_true). If not, this is a development area: a small project or course that you can honestly describe afterwards would address it. Do not list it before you can back it up.")
        elif s.listed_only:
            add("improve_existing", "high", f"Show {s.skill} in action, not only in the skills list",
                f"{s.skill} is required and appears only in your skills list. Add it to the experience or project bullet where you actually applied it so a reviewer can see evidence.",
                example=f"[Action verb] [deliverable] using {s.skill} [and other tools], resulting in [measured outcome, if known].")
    prefs = [x for x in m.skill_matches if x.level == "preferred" and x.status in ("missing", "related")]
    if prefs:
        add("develop", "low", "Preferred skills not evidenced",
            "Nice-to-have skills with no or only related evidence: " + ", ".join(p.skill for p in prefs[:10]) + ". Mention any you genuinely have; otherwise treat them as optional learning goals.")

    # --- 2. terminology alignment -----------------------------------------------------------
    jd_forms: dict[str, set] = {}
    for c, surf, _ in sk.find_skills(jd.text):
        jd_forms.setdefault(c, set()).add(re.sub(r"[\s\-]+", " ", surf.lower()))
    mism = []
    for s in resume.skills:
        forms = {re.sub(r"[\s\-]+", " ", f.lower()) for f in s.surface_forms}
        if s.name in jd_forms and not (forms & jd_forms[s.name]):
            mism.append((sorted(forms)[0], sorted(jd_forms[s.name])[0]))
    if mism:
        add("improve_existing", "low", "Use the job's wording for skills you already have",
            "Automated keyword filters often match exact terms. " + "; ".join(f"you wrote '{a}', the job says '{b}'" for a, b in mism[:5]) + ". Use the job's term where it means the same thing.")

    # --- 3. prioritisation ------------------------------------------------------------------
    if len(m.project_evidence) >= 1 and len(resume.projects) > 1:
        best = m.project_evidence[0]
        if best["relevance"] >= 0.35 and resume.projects[0].name != best["title"]:
            add("improve_existing", "medium", f"Move the project '{best['title']}' to the top",
                "It is your most job-relevant project but is not listed first. Lead with it for this application.",
                evidence=(best["lines"] or [None])[0])
    ordered = [s.name for s in resume.skills if s.listed]
    wanted = {x.skill for x in m.skill_matches if x.status in ("matched", "implied")}
    if len(ordered) >= 6 and len(wanted & set(ordered)) >= 3 and not (set(ordered[:3]) & wanted):
        add("improve_existing", "low", "Reorder your skills section",
            "None of the first three skills listed are ones this job asks for. Put the job-relevant skills you really have first: " + ", ".join(s for s in ordered if s in wanted)[:120] + ".")
    if m.experience_evidence and resume.experience:
        top = m.experience_evidence[0]
        if top["relevance"] >= 0.3 and top["lines"]:
            add("improve_existing", "medium", f"Lead with your most relevant bullets in '{top['title'] or 'your top role'}'",
                "These lines best match the job's requirements; place them first and expand them (scope, tools, outcome):", evidence=" | ".join(top["lines"][:2]))

    # --- 4. clarity and impact of existing content ------------------------------------------
    weak = _weak_bullets(resume)
    if weak:
        add("improve_existing", "medium", f"Strengthen {len(weak)} bullet(s) that lack a clear action",
            "Start with a specific action verb and state your own contribution, the technique/tool and the purpose. Examples from your resume that could be tightened: " + " || ".join(w[:110] for w in weak[:3]),
            example="[Built / Designed / Implemented] [what] using [tools] to [purpose]; [measured result, only if you have one].")
    lines = bullet_lines(resume)
    if lines and (stats.get("quantified_share") or 0) < 0.2:
        add("improve_existing", "medium", "Add real outcomes where you have them",
            "Few bullets state an outcome. Where you know the actual figure (data size, accuracy, users, time saved, cost), add it. If you do not have a measured result, describe scope or impact qualitatively instead; do not estimate numbers.",
            example="… [metric name] improved from [baseline] to [result] (measured by [method]).")
    short_rel = [p for p in resume.projects if len(" ".join(p.description).split()) < 12]
    if short_rel:
        add("improve_existing", "medium", f"Expand thin project descriptions ({len(short_rel)})",
            "Projects with very little description: " + ", ".join(repr(p.name) for p in short_rel[:4]) + ". For each, state the problem, your role, the technologies and the outcome.",
            example="[Project]: solved [problem] by [approach] using [technologies]; [outcome or what you learned].")
    no_tech = [p for p in resume.projects if not p.technologies]
    if no_tech:
        add("add_if_true", "medium", "List the technologies used in each project",
            "No technologies were detected in: " + ", ".join(repr(p.name) for p in no_tech[:4]) + ". Naming the tools you really used makes relevant skills visible.")
    empty = [e for e in resume.experience if not e.highlights]
    if empty:
        add("add_if_true", "medium", "Describe what you did in roles that have no bullets",
            "No responsibilities or achievements were found for: " + ", ".join(f"{e.title or e.organization}" for e in empty[:4]) + ".")
    undated = [e for e in resume.experience if not e.dates.raw]
    if undated:
        add("add_if_true", "medium", "Add dates to experience entries", "Dates are missing for: " + ", ".join(f"{e.title or e.organization}" for e in undated[:4]) + ". Dates are used to judge experience length.")

    # --- 5. sections ----------------------------------------------------------------------
    if not resume.summary:
        add("add_if_true", "low", "Consider a short, role-specific summary",
            "A 2-3 line summary can state your focus and strongest relevant evidence for this role. Use only facts already supported by your resume.",
            example="[Role/field] with [N years or study background] experience in [your real strengths relevant to the job].")
    tech_job = any(sk.CATEGORY.get(r.skill) in ("Programming", "ML/AI", "Data", "Web", "Cloud/DevOps") for r in jd.requirements)
    if tech_job and "github" not in resume.links and "linkedin" not in resume.links:
        add("add_if_true", "low", "Add professional profile links", "This is a technical role; a GitHub, portfolio or LinkedIn link lets reviewers verify your work. Include them only if they are current.")
    if not resume.email:
        add("restructure", "high", "Add a plain-text e-mail address", "No e-mail was detected; recruiters and parsers need one.")
    if resume.extraction_method in ("pdf-ocr", "pdf-text+ocr"):
        add("restructure", "high", "Submit a text-based PDF or DOCX", "This file needed OCR, which can garble text; export from your editor rather than scanning.")
    if not resume.sections_found:
        add("restructure", "high", "Use standard section headings", "No headings such as Experience, Education, Skills or Projects were recognised.")

    # --- 6. experience, education and certifications ---------------------------------------
    if jd.min_years and resume.total_experience_months is not None:
        eff = (resume.total_experience_months + 0.5 * (resume.internship_months or 0)) / 12
        if eff < jd.min_years:
            add("develop", "medium", f"Experience length: ~{eff:.1f} yrs vs {jd.min_years}+ requested",
                "Be accurate about dates. Emphasise relevant depth (scope, ownership, complexity) in projects and internships rather than inflating duration.")
    if m.education.get("applicable") and m.education.get("score", 1) < 0.6:
        add("restructure" if m.education.get("uncertain") else "develop", "medium", "Education requirement",
            m.education["detail"] + (" Make sure degree, institution and years are in a dedicated Education section." if m.education.get("uncertain") else " If you hold a relevant qualification not shown here, add it; otherwise consider how experience may offset it."))
    for c in m.certifications:
        if c["status"] != "matched":
            add("develop" if c["level"] == "required" else "add_if_true", "medium" if c["level"] == "required" else "low", "Certification mentioned in the job",
                f"The job mentions: \"{c['requirement'][:140]}\". " + ("You hold a related credential" + (f" ({c['resume_certification']})" if c["resume_certification"] else "") + ", but not an exact match. " if c["status"] == "related" else "") + "List any relevant certification you already hold (with issuer and year); otherwise it is an optional development goal.")
    recs.sort(key=lambda r: PRIORITY_ORDER[r.priority])
    return recs
