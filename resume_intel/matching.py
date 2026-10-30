"""Hybrid, explainable resume <-> job matching.

Compatibility = weighted blend of seven evidence dimensions (see ``DEFAULT_WEIGHTS``). Dimensions that
do not apply to a given job (e.g. no certification mentioned) are dropped and the remaining weights are
renormalised. The score estimates *job relevance* of the resume content; it is not a hiring probability.
Only job-relevant content is used: name, contact details, links and education dates never enter scoring.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import skills as sk
from .baseline import baseline_match
from .jobs import FIELD_RELATED, JobDescription
from .parser import ParsedResume
from .semantic import Encoder, get_encoder

DEFAULT_WEIGHTS = {
    "skills": 0.30,        # required/preferred skill alignment, weighted by how well each skill is evidenced
    "experience": 0.25,    # relevance of work history (+ years of experience if the job states a minimum)
    "semantic": 0.20,      # embedding similarity between job requirement lines and resume evidence lines
    "projects": 0.10,      # relevance of projects
    "education": 0.05,     # degree level / field vs the job's stated requirement
    "certifications": 0.05,  # certifications named or implied by the job
    "lexical": 0.05,       # TF-IDF cosine (kept small: rewards shared wording, not meaning)
}
LABELS = {"skills": "Skill alignment", "experience": "Experience relevance", "semantic": "Semantic similarity",
          "projects": "Project relevance", "education": "Education alignment", "certifications": "Certifications", "lexical": "Lexical (TF-IDF) similarity"}
LEVEL_WEIGHT = {"required": 1.0, "preferred": 0.5}
LISTED_ONLY_CREDIT = 0.5   # skill appears only in a skills list, with no supporting context (claimed, not demonstrated)
LEXICAL_CAP = 0.35         # TF-IDF cosine at/above which the lexical dimension saturates
EVIDENCE_SECTIONS = {"experience", "projects", "certifications", "summary"}
BANDS = [(75, "Strong match"), (55, "Good match"), (35, "Partial match"), (0, "Weak match")]


@dataclass
class SkillMatch:
    skill: str
    level: str
    status: str  # matched | implied | related | missing
    credit: float
    via: Optional[str] = None
    evidence: list = field(default_factory=list)
    listed_only: bool = False
    uncertain: bool = False  # missing, but the parse was weak so it may simply not have been extracted
    note: str = ""
    custom: bool = False
    alternatives: list = field(default_factory=list)  # other skills the job accepts for this requirement


@dataclass
class Dimension:
    key: str
    label: str
    score: Optional[float]  # 0-1, None when not applicable
    weight: float
    applicable: bool
    detail: str = ""
    contribution: float = 0.0  # points (0-100 scale) added to the overall score


@dataclass
class MatchResult:
    score: float
    band: str
    dimensions: list
    skill_matches: list
    required_coverage: float
    preferred_coverage: float
    experience_evidence: list
    project_evidence: list
    semantic_evidence: list
    education: dict
    certifications: list
    baseline: dict
    encoder: str
    confidence: str
    parse_quality: float
    warnings: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @property
    def missing_required(self):
        return [m for m in self.skill_matches if m.level == "required" and m.status == "missing"]

    @property
    def matched_required(self):
        return [m for m in self.skill_matches if m.level == "required" and m.status in ("matched", "implied")]


def band_for(score: float) -> str:
    for cut, name in BANDS:
        if score >= cut:
            return name
    return BANDS[-1][1]


# ----------------------------------------------------------------------------------- skills
def _skill_matches(resume: ParsedResume, jd: JobDescription, listed_credit: float = LISTED_ONLY_CREDIT) -> list[SkillMatch]:
    recs = {s.name: s for s in resume.skills}
    have = set(recs)
    implied = sk.expand_implied(have)
    weak_parse = resume.parse_quality < 0.6 or "skills" not in resume.sections_found
    content_low = resume.content_text.lower()
    out, seen_groups = [], set()
    for req in jd.requirements:
        group = frozenset([req.skill, *req.alternatives])
        if req.alternatives:
            if (group, req.level) in seen_groups or any(g == group for g, _ in seen_groups):
                continue  # "PyTorch or TensorFlow" is one requirement, not two
            seen_groups.add((group, req.level))
        if req.custom:
            hit = re.search(r"(?<![A-Za-z0-9])" + re.escape(req.skill.lower()) + r"(?![A-Za-z0-9])", content_low)
            out.append(SkillMatch(req.skill, req.level, "matched" if hit else "missing", 1.0 if hit else 0.0,
                                  evidence=[{"section": "text", "text": _snippet(resume.content_text, req.skill)}] if hit else [],
                                  uncertain=bool(not hit and weak_parse), custom=True,
                                  note="" if hit else "Free-text term from the job description, matched literally."))
            continue
        best = None
        for cand in [req.skill, *req.alternatives]:
            if cand in have:
                r = recs[cand]
                evidenced = bool(set(r.sections) & EVIDENCE_SECTIONS)
                credit = 1.0 if evidenced else listed_credit
                m = SkillMatch(req.skill, req.level, "matched", credit, via=cand if cand != req.skill else None, evidence=r.evidence[:2], listed_only=not evidenced,
                               note="Listed in a skills section but not demonstrated in experience or projects." if not evidenced else "")
            elif cand in implied:
                via = implied[cand]
                m = SkillMatch(req.skill, req.level, "implied", sk.IMPLIED_CREDIT, via=via, evidence=recs[via].evidence[:2], note=f"Implied by '{via}' (not stated explicitly).")
            else:
                rel = [x for x in sk.related_skills(cand) if x in have]
                if rel:
                    rel.sort(key=lambda x: (not (set(recs[x].sections) & EVIDENCE_SECTIONS), x))
                    via = rel[0]
                    m = SkillMatch(req.skill, req.level, "related", sk.RELATED_CREDIT, via=via, evidence=recs[via].evidence[:2], note=f"Related skill '{via}' present; '{cand}' itself is not evidenced.")
                else:
                    m = SkillMatch(req.skill, req.level, "missing", 0.0, uncertain=weak_parse, note="No evidence of this skill or a closely related one was found." + (" The resume was only partly parsed, so it may be present but not extracted." if weak_parse else ""))
            if best is None or m.credit > best.credit:
                best = m
        best.alternatives = list(req.alternatives)
        if req.alternatives:
            best.note = (best.note + " " if best.note else "") + f"Job accepts any of: {', '.join(sorted(group))}."
        out.append(best)
    return out


def _snippet(text: str, term: str, width: int = 90) -> str:
    i = text.lower().find(term.lower())
    if i < 0:
        return ""
    return text[max(0, i - width // 2): i + len(term) + width // 2].replace("\n", " ").strip()


def _skill_score(matches: list[SkillMatch]) -> Optional[float]:
    if not matches:
        return None
    tot = sum(LEVEL_WEIGHT[m.level] for m in matches)
    return sum(LEVEL_WEIGHT[m.level] * m.credit for m in matches) / tot


# ----------------------------------------------------------------------------------- evidence pool
def _build_pool(resume: ParsedResume):
    pool = []  # dicts: section, entry, text (display), enc (text sent to the encoder)
    for i, e in enumerate(resume.experience):
        ctx = " ".join(x for x in (e.title,) if x)
        lines = e.highlights or [", ".join(x for x in (e.title, e.organization) if x)]
        for ln in lines:
            if ln.strip():
                pool.append({"section": "experience", "entry": i, "text": ln, "enc": f"{ctx}: {ln}" if ctx else ln})
    for i, p in enumerate(resume.projects):
        ctx = p.name or "Project"
        for ln in p.description:
            pool.append({"section": "projects", "entry": i, "text": ln, "enc": f"{ctx}: {ln}"})
        if p.technologies:
            pool.append({"section": "projects", "entry": i, "text": f"{ctx} — technologies: {', '.join(p.technologies)}", "enc": f"{ctx}: uses {', '.join(p.technologies)}"})
        if not p.description and p.name and not p.technologies:
            pool.append({"section": "projects", "entry": i, "text": p.name, "enc": p.name})
    if resume.summary:
        pool.append({"section": "summary", "entry": None, "text": resume.summary[:400], "enc": resume.summary[:400]})
    listed = [s.name for s in resume.skills if s.listed]
    if listed:
        pool.append({"section": "skills", "entry": None, "text": "Skills: " + ", ".join(listed), "enc": "Skills: " + ", ".join(listed)})
    for c in resume.certifications:
        pool.append({"section": "certifications", "entry": None, "text": c.name, "enc": c.name + (f" ({c.issuer})" if c.issuer else "")})
    for e in resume.education:
        t = " ".join(x for x in (e.degree, "in " + e.field_of_study if e.field_of_study else None) if x)
        if t:
            pool.append({"section": "education", "entry": None, "text": t, "enc": t})
    return pool


def _entry_scores(kind: str, entries, pool, S, req_w, jd: JobDescription, resume: ParsedResume):
    """Relevance of each experience/project entry: 0.5 * semantic + 0.5 * skill coverage."""
    jd_skills = [(r.skill, LEVEL_WEIGHT[r.level]) for r in jd.requirements if not r.custom]
    jd_total = sum(w for _, w in jd_skills)
    out = []
    for i, e in enumerate(entries):
        idx = [j for j, p in enumerate(pool) if p["section"] == kind and p["entry"] == i]
        sim = float((S[:, idx].max(axis=1) * req_w).sum() / req_w.sum()) if idx and S.size else 0.0
        if kind == "experience":
            have = set(e.skills)
            blob = " ".join(e.highlights)
        else:
            have = set(e.technologies)
            blob = " ".join(e.description)
        have |= {c for c, _, _ in sk.find_skills(blob)}
        have |= set(sk.expand_implied(have))
        cov = sum(w for s_, w in jd_skills if s_ in have) / jd_total if jd_total else 0.0
        matched = [s_ for s_, _ in jd_skills if s_ in have]
        # best supporting lines
        lines = sorted(((float(S[:, j].max()) if S.size else 0.0, pool[j]["text"]) for j in idx), reverse=True)[:2]
        out.append({"index": i, "relevance": 0.5 * sim + 0.5 * cov, "semantic": sim, "skill_coverage": cov, "matched_skills": matched, "lines": [t for _, t in lines]})
    return out


def _agg_entries(scores: list[dict]) -> Optional[float]:
    if not scores:
        return None
    rel = sorted((s["relevance"] for s in scores), reverse=True)
    return 0.5 * rel[0] + 0.5 * (sum(rel[:3]) / len(rel[:3]))


# ----------------------------------------------------------------------------------- other dims
def _education(resume: ParsedResume, jd: JobDescription):
    if jd.education_level is None:
        return None, {"applicable": False, "detail": "The job description states no degree requirement."}
    levels = [e.level for e in resume.education if e.level]
    if not resume.education:
        return 0.0, {"applicable": True, "uncertain": True, "required_level": jd.education_level, "resume_level": None,
                     "detail": "No education was detected in the resume, so this may be a missing section or an extraction gap rather than a true gap."}
    top = max(levels) if levels else None
    if top is None:
        lvl_score, d = 0.0, "Education entries were found but no recognisable degree level."
    elif top >= jd.education_level:
        lvl_score, d = 1.0, "Highest detected degree meets or exceeds the stated requirement."
    elif top == jd.education_level - 1:
        lvl_score, d = 0.5, "Highest detected degree is one level below the stated requirement."
    else:
        lvl_score, d = 0.2, "Highest detected degree is below the stated requirement."
    if jd.education_flexible and lvl_score < 1.0:
        lvl_score = max(lvl_score, 0.5)
        d += " The job accepts equivalent experience."
    score, field_note = lvl_score, ""
    if jd.education_fields:
        fields = " ".join(((e.field_of_study or "") + " " + (e.degree or "")).lower() for e in resume.education)
        if not fields.strip():
            fscore, field_note = 0.5, " Field of study could not be determined (neutral credit)."
        elif any(f in fields for f in jd.education_fields):
            fscore, field_note = 1.0, " Field of study matches."
        elif any(any(rel in fields for rel in FIELD_RELATED.get(f, [])) for f in jd.education_fields):
            fscore, field_note = 0.6, " Field of study is related."
        else:
            fscore, field_note = 0.0, " Field of study is not among those named."
        score = 0.7 * lvl_score + 0.3 * fscore
    return score, {"applicable": True, "uncertain": False, "required_level": jd.education_level, "resume_level": top, "detail": d + field_note}


_CERT_STOP = set("certified certification certifications certificate certificates preferred required plus such like eg ie relevant professional associate or and the a an in of with as for to is are".split())


def _cert_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9+#.]{1,}", text.lower()) if w not in _CERT_STOP and len(w) > 1}


def _certs(resume: ParsedResume, jd: JobDescription):
    if not jd.certifications:
        return None, [], False
    res_blobs = [(c, (c.name + " " + (c.issuer or "")).lower()) for c in resume.certifications]
    rows, num, den = [], 0.0, 0.0
    for req in jd.certifications:
        toks = _cert_tokens(req["text"])
        jd_sk = sk.skill_set(req["text"])
        best, credit = None, 0.0
        for c, blob in res_blobs:
            ctoks = _cert_tokens(blob)
            overlap = len(toks & ctoks) / max(1, len(toks))
            has_sk = bool(jd_sk & sk.skill_set(blob)) or any(s_ in blob for s_ in jd_sk)
            cr = 1.0 if overlap >= 0.5 else 0.5 if (has_sk or overlap >= 0.25) else 0.0
            if cr > credit:
                best, credit = c, cr
        w = LEVEL_WEIGHT[req["level"]]
        num += w * credit
        den += w
        rows.append({"requirement": req["text"], "level": req["level"], "status": "matched" if credit == 1.0 else "related" if credit else "missing",
                     "credit": credit, "resume_certification": best.name if best else None})
    return num / den, rows, not resume.certifications and "certifications" not in resume.sections_found


# ----------------------------------------------------------------------------------- main
def match(resume: ParsedResume, jd: JobDescription, encoder: Optional[Encoder] = None, weights: Optional[dict] = None, listed_credit: Optional[float] = None, projects_na_if_none: bool = True) -> MatchResult:
    enc = encoder or get_encoder()
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    warnings = list(jd.warnings)

    lc = LISTED_ONLY_CREDIT if listed_credit is None else listed_credit
    smatches = _skill_matches(resume, jd, lc)
    skill_score = _skill_score(smatches)
    req_m = [m for m in smatches if m.level == "required"]
    pref_m = [m for m in smatches if m.level == "preferred"]
    req_cov = sum(m.status in ('matched', 'implied') for m in req_m) / len(req_m) if req_m else 1.0
    pref_cov = sum(m.status in ('matched', 'implied') for m in pref_m) / len(pref_m) if pref_m else 1.0

    # lexical
    try:
        v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = v.fit_transform([resume.content_text or resume.text, jd.text])
        cos = float(cosine_similarity(X[0:1], X[1:2])[0, 0])
    except ValueError:
        cos = 0.0
    lexical = min(1.0, cos / LEXICAL_CAP)

    # semantic
    pool = _build_pool(resume)
    reqs = [r for r in jd.requirement_lines if len(r["text"].split()) >= 3]
    req_w = np.array([LEVEL_WEIGHT[r["level"]] for r in reqs]) if reqs else np.zeros(0)
    sem_evidence, S = [], np.zeros((len(reqs), len(pool)))
    if reqs and pool:
        S = enc.scale(enc.similarity([r["text"] for r in reqs], [p["enc"] for p in pool]))
        best_idx = S.argmax(axis=1)
        sem_score = float((S.max(axis=1) * req_w).sum() / req_w.sum())
        for i, r in enumerate(reqs):
            j = int(best_idx[i])
            sem_evidence.append({"requirement": r["text"], "level": r["level"], "similarity": round(float(S[i, j]), 3), "best_evidence": pool[j]["text"], "section": pool[j]["section"]})
        sem_evidence.sort(key=lambda d: -d["similarity"])
    else:
        sem_score = 0.0
        if not pool:
            warnings.append("No resume evidence lines were available for semantic comparison.")

    # experience / projects
    exp_sc = _entry_scores("experience", resume.experience, pool, S, req_w if len(req_w) else np.ones(1), jd, resume) if resume.experience else []
    proj_sc = _entry_scores("projects", resume.projects, pool, S, req_w if len(req_w) else np.ones(1), jd, resume) if resume.projects else []
    exp_rel = _agg_entries(exp_sc)
    exp_detail = ""
    eff_months = None
    if resume.total_experience_months is not None:
        eff_months = (resume.total_experience_months or 0) + 0.5 * (resume.internship_months or 0)
    if exp_rel is None:
        exp_score = 0.0
        exp_detail = "No work-experience entries were extracted" + (" (no experience section found, so this may be an extraction gap or a fresher profile)." if "experience" not in resume.sections_found else ".")
    else:
        exp_score = exp_rel
        exp_detail = f"Mean of best and top-3 entry relevance = {exp_rel:.2f}."
        if jd.min_years:
            if eff_months is None:
                exp_detail += f" The job asks for {jd.min_years}+ years but no usable dates were extracted, so years were not scored."
            else:
                fit = min(1.0, eff_months / (12 * jd.min_years))
                exp_score = 0.75 * exp_rel + 0.25 * fit
                exp_detail += f" Estimated {eff_months / 12:.1f} yrs (internships at 50%) vs {jd.min_years}+ required -> years-fit {fit:.2f}."
    proj_rel = _agg_entries(proj_sc)
    proj_score = proj_rel if proj_rel is not None else 0.0
    proj_detail = f"Mean of best and top-3 project relevance = {proj_rel:.2f}." if proj_rel is not None else "No projects were extracted."

    edu_score, edu = _education(resume, jd)
    cert_score, cert_rows, cert_uncertain = _certs(resume, jd)

    raw = {
        "skills": (skill_score, bool(smatches), f"{sum(m.status in ('matched', 'implied') for m in smatches)}/{len(smatches)} job skills found; weights required=1.0, preferred=0.5; partial credit for implied ({sk.IMPLIED_CREDIT}) / related ({sk.RELATED_CREDIT}) / list-only ({lc}) evidence."),
        "experience": (exp_score, True, exp_detail),
        "semantic": (sem_score, bool(reqs), f"Average best-match similarity of {len(reqs)} job requirement lines to resume evidence, using {enc.name}."),
        "projects": (proj_score, bool(resume.projects) or not projects_na_if_none, proj_detail if resume.projects or not projects_na_if_none else "No projects were listed, so this factor is not applied (many professionals list none); experience carries the evidence."),
        "education": (edu_score, edu["applicable"], edu["detail"]),
        "certifications": (cert_score, cert_score is not None, "Certifications named in the job vs those found in the resume." if cert_score is not None else "The job names no certifications."),
        "lexical": (lexical, True, f"TF-IDF cosine {cos:.3f}; saturates at {LEXICAL_CAP}."),
    }
    total_w = sum(w[k] for k, (s_, ok, _) in raw.items() if ok)
    dims, score = [], 0.0
    for k, (s_, ok, detail) in raw.items():
        wn = w[k] / total_w if ok and total_w else 0.0
        contrib = 100 * wn * (s_ or 0.0) if ok else 0.0
        score += contrib
        dims.append(Dimension(k, LABELS[k], round(s_, 4) if ok and s_ is not None else None, round(wn, 4), ok, detail, round(contrib, 2)))
    score = round(score, 2)

    pq = resume.parse_quality
    conf = "high" if pq >= 0.7 else "medium" if pq >= 0.5 else "low"
    if conf != "high":
        warnings.append(f"Resume parse quality is {conf}: some missing items may be extraction gaps. Review manually.")
    if enc.name.startswith("concept"):
        warnings.append("Semantic similarity used the lightweight concept-hash fallback (the MiniLM model is not available locally).")

    def evid(kind, entries, scs):
        rows = []
        for s_ in sorted(scs, key=lambda x: -x["relevance"])[:3]:
            e = entries[s_["index"]]
            if kind == "experience":
                head = {"title": e.title, "organization": e.organization, "dates": e.dates.raw}
            else:
                head = {"title": e.name, "organization": None, "dates": ""}
            rows.append({**head, "relevance": round(s_["relevance"], 3), "matched_skills": s_["matched_skills"], "lines": s_["lines"]})
        return rows

    edu_out = {k: v for k, v in edu.items()}
    edu_out["score"] = edu_score
    return MatchResult(
        score=score, band=band_for(score), dimensions=dims, skill_matches=smatches, required_coverage=round(req_cov, 3), preferred_coverage=round(pref_cov, 3),
        experience_evidence=evid("experience", resume.experience, exp_sc), project_evidence=evid("projects", resume.projects, proj_sc),
        semantic_evidence=sem_evidence[:8], education=edu_out, certifications=cert_rows,
        baseline=baseline_match(resume.text, jd.text), encoder=enc.name, confidence=conf, parse_quality=pq, warnings=warnings,
        weights={k: round(v_, 4) for k, v_ in w.items()},
    )
