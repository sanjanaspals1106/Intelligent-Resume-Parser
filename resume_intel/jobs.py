"""Job-description parsing: required vs preferred skills, years, education, certifications."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import skills as sk

REQUIRED_HEAD = re.compile(r"requirement|qualification|must[\s-]*have|what you(?:'| a)?(?:ll)? (?:need|bring|have)|what we(?:'re| are) looking|who you are|your profile|you have|skills required|required skills|looking for|minimum|basic qual|about you|essential|key skills|technical skills|skills and experience|skills & experience", re.I)
PREFERRED_HEAD = re.compile(r"prefer|nice[\s-]*to[\s-]*have|bonus|good[\s-]*to[\s-]*have|desired|desirable|\bplus\b|extra credit|additional (?:qualification|skill)|advantage", re.I)
RESP_HEAD = re.compile(r"responsibilit|what you(?:'| wi)?ll do|what you will do|duties|your role|the role|day[\s-]*to[\s-]*day|you will|key tasks|about the (?:role|job|position)|job description|role overview|what you.ll be doing", re.I)
IGNORE_HEAD = re.compile(r"benefit|perks|about (?:us|the company|the team)|who we are|we offer|why join|compensation|salary|equal opportunity|how to apply|our (?:mission|culture|values)|what we offer|company", re.I)
PREFERRED_CUE = re.compile(r"\b(preferred|nice[\s-]*to[\s-]*have|a plus|is a plus|are a plus|bonus|good[\s-]*to[\s-]*have|desirable|desired|ideally|an? advantage|exposure to|optional|would be (?:great|an advantage|nice))\b", re.I)
REQUIRED_CUE = re.compile(r"\b(required|must|minimum|essential|mandatory|need to|needs to)\b", re.I)
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:(?:-|–|to)\s*(\d{1,2})\s*\+?\s*)?(?:years?|yrs?)\b", re.I)
EDU_LEVELS = [
    (re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.I), 5),
    (re.compile(r"\b(master'?s?|m\.?sc|m\.?tech|mba|m\.?s\.?\b|graduate degree|postgraduate)\b", re.I), 4),
    (re.compile(r"\b(bachelor'?s?|b\.?sc|b\.?tech|b\.?e\.?\b|b\.?s\.?\b|undergraduate|university degree|college degree)\b", re.I), 3),
    (re.compile(r"\b(associate'?s? degree|diploma)\b", re.I), 2),
]
FIELD_KEYWORDS = ["computer science", "software engineering", "information technology", "information systems", "data science", "statistics", "mathematics", "applied mathematics", "electrical engineering", "electronics", "engineering", "physics", "business", "economics", "finance", "artificial intelligence", "machine learning", "analytics", "operations research", "design", "cybersecurity"]
FIELD_RELATED = {
    "computer science": ["software engineering", "information technology", "information systems", "data science", "artificial intelligence", "computer", "computing"],
    "software engineering": ["computer science", "information technology", "computer"],
    "information technology": ["computer science", "information systems", "software engineering", "computer"],
    "data science": ["computer science", "statistics", "mathematics", "analytics", "artificial intelligence"],
    "statistics": ["mathematics", "data science", "analytics", "applied mathematics"],
    "mathematics": ["statistics", "applied mathematics", "data science", "physics"],
    "engineering": ["computer", "electrical", "electronics", "software", "information technology", "mechanical", "civil"],
}
CUSTOM_STOP = set("""experience years year degree ability strong excellent knowledge understanding skills communication team teams work working good bachelor master phd required preferred plus bonus field related equivalent proven demonstrated track record passion environment fast paced english written verbal problem solving problem-solving collaboration self motivated motivated candidate role responsibilities""".split())
ALT_JOINER = re.compile(r"^\s*(?:/|\bor\b|\band/or\b|,\s*or\b|\s+or\b)\s*$", re.I)


@dataclass
class Requirement:
    skill: str
    level: str  # "required" | "preferred"
    basis: str  # why this level was assigned
    snippet: str = ""
    alternatives: list = field(default_factory=list)  # other skills that also satisfy this requirement
    custom: bool = False  # free-text term outside the skill taxonomy
    category: str = ""


@dataclass
class JobDescription:
    text: str
    title: Optional[str] = None
    requirements: list = field(default_factory=list)
    requirement_lines: list = field(default_factory=list)  # [{"text":..,"level":..}]
    min_years: Optional[int] = None
    education_level: Optional[int] = None
    education_fields: list = field(default_factory=list)
    education_flexible: bool = False  # "or equivalent experience"
    certifications: list = field(default_factory=list)  # [{"text":..,"level":..}]
    structured: bool = False
    warnings: list = field(default_factory=list)

    @property
    def required_skills(self):
        return [r.skill for r in self.requirements if r.level == "required"]

    @property
    def preferred_skills(self):
        return [r.skill for r in self.requirements if r.level == "preferred"]

    def to_dict(self):
        d = asdict(self)
        d.pop("text")
        return d


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z&' ]", " ", line)).strip().lower()


def _classify_heading(line: str) -> Optional[str]:
    t = line.strip()
    if not t or len(t) > 70 or len(t.split()) > 9:
        return None
    if t.endswith((".", ",", ";")):
        return None
    if re.match(r"^\s*(?:[•●○◦▪*·-]|\d+[.)])\s", t):
        return None
    if not (t.endswith(":") or t.isupper() or t.istitle() or len(t.split()) <= 4):
        return None
    n = _norm(t)
    if PREFERRED_HEAD.search(n):
        return "preferred"
    if IGNORE_HEAD.search(n):
        return "ignore"
    if RESP_HEAD.search(n):
        return "responsibilities"
    if REQUIRED_HEAD.search(n):
        return "required"
    return None


def _strip_bullet(l: str) -> str:
    return re.sub(r"^\s*(?:[•●○◦▪▫■➢➤►*·–—-]|\d{1,2}[.)])\s+", "", l).strip()


def _alt_groups(line: str) -> dict:
    """Map skill -> list of alternative skills joined by "or" / "/" in ``line``."""
    spans = []
    for c, surf, o in sk.find_skills(line):
        spans.append((o, o + len(surf), c))
    spans.sort()
    groups, cur = {}, []
    for i, (s, e, c) in enumerate(spans):
        joiner = line[spans[i - 1][1]:s] if i else ""
        # "A / B" only means "either" when both skills belong to the same family (PyTorch/TensorFlow)
        if i and ALT_JOINER.match(joiner) and ("/" not in joiner or c in sk.related_skills(spans[i - 1][2])):
            cur.append(c)
        else:
            if len(cur) > 1:
                for x in cur:
                    groups[x] = [y for y in cur if y != x]
            cur = [c]
    if len(cur) > 1:
        for x in cur:
            groups[x] = [y for y in cur if y != x]
    return groups


def parse_job_description(text: str) -> JobDescription:
    jd = JobDescription(text=text)
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n")]
    # title
    for l in lines[:5]:
        m = re.match(r"^\s*(?:job\s*title|position|role)\s*[:\-]\s*(.+)$", l, re.I)
        if m:
            jd.title = m.group(1).strip()
            break
    if not jd.title:
        for l in lines:
            if l.strip():
                if len(l.split()) <= 10 and not l.strip().endswith("."):
                    jd.title = l.strip(" :")
                break

    # sectioning
    section, tagged = None, []
    for l in lines:
        if not l.strip():
            continue
        h = _classify_heading(l)
        if h:
            section = h
            jd.structured = jd.structured or h in ("required", "preferred")
            continue
        tagged.append((section, l))
    has_req_section = any(s == "required" for s, _ in tagged)

    reqs: dict[str, Requirement] = {}
    custom: dict[str, Requirement] = {}
    rank = {"preferred": 0, "required": 1}

    def put(skill, level, basis, snippet, alts, is_custom=False):
        store = custom if is_custom else reqs
        cur = store.get(skill)
        if cur is None or rank[level] > rank[cur.level]:
            store[skill] = Requirement(skill, level, basis, snippet[:200], alts, is_custom, sk.CATEGORY.get(skill, "Other (free-text)"))

    for section, l in tagged:
        if section == "ignore":
            continue
        line = _strip_bullet(l)
        # decide level for this line
        if PREFERRED_CUE.search(line):
            level, basis = "preferred", "line contains a preferred/nice-to-have cue"
        elif section == "preferred":
            level, basis = "preferred", "listed under a preferred/nice-to-have heading"
        elif section == "required":
            level, basis = "required", "listed under a requirements heading"
        elif section == "responsibilities":
            if has_req_section:
                level, basis = "preferred", "mentioned only in responsibilities, not in the requirements list"
            else:
                level, basis = "required", "mentioned in responsibilities (no requirements section found)"
        elif REQUIRED_CUE.search(line):
            level, basis = "required", "line contains a required/must cue"
        else:
            level, basis = ("preferred", "mentioned outside requirements") if has_req_section else ("required", "no requirement structure found; treated as required")
        alts = _alt_groups(line)
        for canon in {c for c, _, _ in sk.find_skills(line)}:
            put(canon, level, basis, line, alts.get(canon, []))
        if section in ("required", "preferred", None) and len(line.split()) <= 120:
            jd.requirement_lines.append({"text": line, "level": level})
        elif section == "responsibilities":
            jd.requirement_lines.append({"text": line, "level": "required" if level == "required" else "preferred"})

        # years of experience
        if re.search(r"experience|exp\b", line, re.I) and level == "required":
            for m in YEARS_RE.finditer(line):
                n = int(m.group(1))
                if n <= 30:
                    jd.min_years = max(jd.min_years or 0, n)
        # education
        if re.search(r"degree|bachelor|master|ph\.?d|b\.?tech|m\.?tech|diploma|graduate|b\.?sc|m\.?sc|b\.?e\b", line, re.I):
            lvls = [lv for rx, lv in EDU_LEVELS if rx.search(line)]
            if not lvls and re.search(r"\bdegree\b", line, re.I):
                lvls = [3]  # a bare "degree" is read as bachelor's level
            if lvls and level == "required":
                lv = min(lvls)
                jd.education_level = lv if jd.education_level is None else min(jd.education_level, lv)
                low = line.lower()
                for f in FIELD_KEYWORDS:
                    if f in low and f not in jd.education_fields:
                        jd.education_fields.append(f)
                if re.search(r"equivalent (?:practical )?experience|or equivalent|or related experience", line, re.I):
                    jd.education_flexible = True
        # certifications
        if re.search(r"certif|licen[cs]e", line, re.I):
            jd.certifications.append({"text": line[:200], "level": level})
        # free-text terms (outside taxonomy): short bullets / "Label: a, b, c" lists
        cands = []
        body = line.split(":", 1)[1] if ":" in line and len(line.split(":")[0].split()) <= 4 else line
        parts = [p.strip(" .()") for p in re.split(r"\s*[,;/•|]\s*|\s+and\s+|\s+or\s+", body)]
        if section in ("required", "preferred") and (len(parts) >= 3 or len(line.split()) <= 4):
            cands = parts
        for c in cands:
            w = c.split()
            if 1 <= len(w) <= 3 and 2 <= len(c) <= 30 and c[:1].isupper() and not sk.skill_set(c) and not (set(x.lower() for x in w) & CUSTOM_STOP) and not re.search(r"\d", c):
                put(c, level, basis + " (free-text term, matched literally)", line, [], True)

    jd.requirements = sorted(reqs.values(), key=lambda r: (r.level != "required", r.skill)) + sorted(custom.values(), key=lambda r: (r.level != "required", r.skill))
    # a skill listed as an alternative of a required one inherits grouping only; dedupe symmetric groups
    if not jd.requirement_lines:
        jd.requirement_lines = [{"text": s.strip(), "level": "required"} for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(s.split()) >= 4][:30]
    jd.requirement_lines = jd.requirement_lines[:40]
    if not jd.requirements:
        jd.warnings.append("No recognisable skills were found in the job description; matching will rely on text similarity only.")
    if not jd.structured:
        jd.warnings.append("The job description has no clear requirements/preferred sections; all mentioned skills were treated as required unless a cue (e.g. 'nice to have') said otherwise.")
    return jd
