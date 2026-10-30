"""Rule-based structured resume parser.

Every extracted item carries a ``confidence`` (high / medium / low). Values that were
guessed rather than read directly are flagged ``inferred``. Fields that could not be
found are left empty/None -- never invented -- and listed in ``warnings``.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

from . import skills as sk

# ----------------------------------------------------------------------------- constants
SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "professional profile", "objective", "career objective", "about me", "about", "career summary", "executive summary", "personal statement"],
    "experience": ["experience", "work experience", "professional experience", "employment", "employment history", "work history", "career history", "internships", "internship experience", "relevant experience", "industry experience", "experience summary", "professional background"],
    "education": ["education", "academic background", "academic qualifications", "qualifications", "education and training", "academics", "educational qualifications", "academic profile", "education and qualifications"],
    "skills": ["skills", "technical skills", "key skills", "core competencies", "technical proficiency", "skills and tools", "tools and technologies", "competencies", "areas of expertise", "technical expertise", "skills and technologies", "core skills", "skill set", "technical summary"],
    "projects": ["projects", "personal projects", "academic projects", "selected projects", "key projects", "project experience", "notable projects", "side projects", "project work", "selected work", "work samples", "portfolio", "open source", "open source contributions"],
    "certifications": ["certifications", "certificates", "licenses", "licenses and certifications", "courses", "training", "professional certifications", "certifications and courses", "courses and certifications", "certification", "training and certifications"],
    "awards": ["awards", "achievements", "honors", "honours", "accomplishments", "awards and honors", "awards and achievements", "honors and awards"],
    "publications": ["publications", "research", "papers", "research publications"],
    "languages": ["languages", "language skills"],
    "interests": ["interests", "hobbies", "hobbies and interests", "extracurricular activities", "extracurriculars", "volunteering", "volunteer experience", "activities", "leadership and activities"],
    "references": ["references"],
}
_HEADING_LOOKUP = {a: sec for sec, al in SECTION_ALIASES.items() for a in al}

_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_YEAR = r"(?:19|20)\d{2}"
_DATE = rf"(?:(?:{_MONTH})\.?,?\s+{_YEAR}|\d{{1,2}}/{_YEAR}|{_YEAR}|present|current|currently|now|ongoing|till\s+date|to\s+date)"
DATE_RANGE_RE = re.compile(rf"(?P<s>{_DATE})\s*(?:-|–|—|‒|to|until|till)\s*(?P<e>{_DATE})", re.I)
SINGLE_DATE_RE = re.compile(rf"\b{_DATE}\b", re.I)
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

EMAIL_RE = re.compile(r"[\w.%+\-]+@[\w\-]+(?:\.[\w\-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\w/.-])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,5}[\s.-]?\d{3,5}[\s.-]?\d{3,5}(?![\w/-])")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\],;|]+|(?:linkedin\.com|github\.com|gitlab\.com|bitbucket\.org|kaggle\.com|medium\.com|leetcode\.com|behance\.net|huggingface\.co)/[^\s<>()\[\],;|]+", re.I)
BULLET_RE = re.compile(r"^\s*(?:[•●○◦▪▫■□➢➤►▶◆◇*·–—-]|\d{1,2}[.)])\s+(?=\S)")
BULLET_ONLY_RE = re.compile(r"^\s*[•●○◦▪▫■□➢➤►▶◆◇*·]\s*")

TITLE_RE = re.compile(r"\b(engineer|developer|analyst|manager|intern|internship|scientist|consultant|lead|architect|designer|associate|researcher|director|administrator|specialist|officer|assistant|trainee|head|programmer|tester|executive|coordinator|technician|professor|lecturer|instructor|teacher|founder|co-founder|president|vice president|vp|supervisor|representative|apprentice|fellow|freelance|freelancer|sde|swe|qa|devops|sre|cto|ceo|statistician|strategist|advisor|mentor|volunteer|writer|editor|accountant|auditor|recruiter|nurse|operator)\b", re.I)
ORG_RE = re.compile(r"\b(inc|ltd|llc|llp|corp|corporation|company|co\.|technologies|technology|solutions|labs|systems|pvt|private|limited|bank|group|software|services|consulting|industries|enterprises|university|institute|hospital|studio|studios|networks|analytics|digital|ventures|foundation|agency|partners|international|global|gmbh|ag|plc|infotech|infosystems|innovations|research)\b", re.I)
INSTITUTION_RE = re.compile(r"\b(university|college|institute|school|academy|bootcamp|polytechnic|iit|nit|iiit|iisc|bits|mit|universit[a-zà-ÿ]*|universidad|hochschule|vidyalaya|campus|faculty)\b", re.I)

DEGREE_PATTERNS = [
    # (regex, level, canonical label) -- ordered; first match wins at a given position
    (r"\b(?:ph\.?\s?d\.?|doctorate|doctor of philosophy|d\.?phil)\b", 5, "PhD"),
    (r"\b(?:master(?:'?s)?(?:\s+of\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|\s+degree|\s+in)?|m\.?\s?tech|m\.?\s?sc\.?|m\.?\s?eng|mba|mca|m\.?\s?s\.?(?=\s*(?:in|\(|,|-|$))|m\.?\s?e\.?(?=\s*(?:in|\(|,|-|$)))", 4, "Master's"),
    (r"\b(?:bachelor(?:'?s)?(?:\s+of\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|\s+degree|\s+in)?|b\.?\s?tech|b\.?\s?sc\.?|b\.?\s?eng|bca|bba|b\.?\s?e\.?(?=\s*(?:in|\(|,|-|$))|b\.?\s?s\.?(?=\s*(?:in|\(|,|-|$))|b\.?\s?a\.?(?=\s*(?:in|\(|,|-|$)))", 3, "Bachelor's"),
    (r"\b(?:associate(?:'?s)?\s+(?:degree|of)|a\.?a\.?s\.?)\b", 2, "Associate"),
    (r"\b(?:diploma|pgdm|post\s?graduate diploma)\b", 2, "Diploma"),
    (r"\b(?:bootcamp|certificate)\b", 2, "Certificate"),
    (r"\b(?:higher secondary|senior secondary|high school|hsc|ssc|12th|10th|intermediate|a-levels?|secondary school)\b", 1, "Secondary"),
]
_DEGREE_RES = [(re.compile(p, re.I), lvl, lab) for p, lvl, lab in DEGREE_PATTERNS]
# Case-sensitive abbreviations that would be ambiguous when case-folded (BE, ME, MS, BS, BA, MA)
_ABBREV_RE = re.compile(r"(?<![A-Za-z])(B\.?E\.?|B\.?S\.?|B\.?A\.?|M\.?E\.?|M\.?S\.?|M\.?A\.?|BE|ME|MS|BS|BA|MA)(?![A-Za-z])")
_ABBREV_LEVEL = {"B": 3, "M": 4}

KNOWN_ISSUERS = ["aws", "amazon web services", "amazon", "google", "microsoft", "coursera", "udemy", "oracle", "cisco", "ibm", "deeplearning.ai", "deeplearning", "pmi", "red hat", "nvidia", "meta", "databricks", "salesforce", "comptia", "edx", "stanford", "mit", "harvard", "linkedin", "kaggle", "hackerrank", "udacity", "infosys", "nptel", "scrum alliance", "isaca", "(isc)", "hashicorp", "snowflake", "tableau", "sas", "datacamp", "pluralsight", "cloudera", "mongodb", "the linux foundation", "linux foundation", "cncf", "unity", "adobe", "duke", "johns hopkins", "university of michigan"]
CERT_WORD_RE = re.compile(r"\b(certified|certificate|certification|specialization|nanodegree|professional certificate|license|associate|practitioner|foundations?|fundamentals|course|bootcamp|masterclass)\b", re.I)
ACTION_VERBS = set("""built developed designed implemented created led managed improved reduced increased optimized optimised automated deployed delivered launched analyzed analysed architected engineered trained fine-tuned evaluated migrated integrated established collaborated coordinated mentored maintained refactored resolved streamlined spearheaded drove achieved authored published presented researched wrote conducted performed supported ran owned executed scaled secured tested configured produced generated extracted processed applied initiated organized organised handled prepared monitored debugged enhanced simplified accelerated modernized contributed revamped crafted constructed formulated validated""".split())
RESULT_RE = re.compile(r"(\d+(?:\.\d+)?\s?(?:%|x\b|k\b|m\b|ms\b|hrs?\b|hours?\b|users?\b|customers?\b|records\b|requests\b)|\$\s?\d|₹\s?\d|\b\d{2,}\+?\b|\b(?:won|winner|ranked|rank|award|awarded|first place|finalist|top \d+|published|patent)\b)", re.I)
GPA_RE = re.compile(r"\b(?:cgpa|gpa|percentage|grade|score)\s*[:\-]?\s*(\d+(?:\.\d+)?\s*(?:/\s*\d+(?:\.\d+)?|%)?)|\b(\d(?:\.\d{1,2})?\s*/\s*(?:4(?:\.0)?|10(?:\.0)?))\b", re.I)


# ----------------------------------------------------------------------------- data model
@dataclass
class DateSpan:
    raw: str = ""
    start: Optional[str] = None  # "YYYY-MM"
    end: Optional[str] = None  # "YYYY-MM" or "present"
    months: Optional[int] = None


@dataclass
class ExperienceEntry:
    title: Optional[str] = None
    organization: Optional[str] = None
    dates: DateSpan = field(default_factory=DateSpan)
    highlights: list = field(default_factory=list)
    skills: list = field(default_factory=list)
    is_internship: bool = False
    confidence: str = "low"
    raw: str = ""


@dataclass
class EducationEntry:
    degree: Optional[str] = None
    level: Optional[int] = None  # 1 secondary .. 5 doctorate
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    dates: DateSpan = field(default_factory=DateSpan)
    grade: Optional[str] = None
    confidence: str = "low"
    raw: str = ""


@dataclass
class ProjectEntry:
    name: Optional[str] = None
    description: list = field(default_factory=list)
    technologies: list = field(default_factory=list)
    achievements: list = field(default_factory=list)
    links: list = field(default_factory=list)
    confidence: str = "low"
    raw: str = ""


@dataclass
class CertificationEntry:
    name: str = ""
    issuer: Optional[str] = None
    year: Optional[str] = None
    confidence: str = "low"


@dataclass
class SkillRecord:
    name: str
    category: str
    listed: bool  # appears in an explicit skills section
    sections: list = field(default_factory=list)
    surface_forms: list = field(default_factory=list)
    evidence: list = field(default_factory=list)  # [{"section":..., "text":...}]


@dataclass
class ParsedResume:
    name: Optional[str] = None
    name_confidence: str = "low"
    email: Optional[str] = None
    phone: Optional[str] = None
    links: dict = field(default_factory=dict)  # {"linkedin": url, "github": url, "other": [urls]}
    summary: str = ""
    skills: list = field(default_factory=list)
    other_listed_skills: list = field(default_factory=list)  # listed but outside the taxonomy
    education: list = field(default_factory=list)
    experience: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    certifications: list = field(default_factory=list)
    awards: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    publications: list = field(default_factory=list)
    sections_found: list = field(default_factory=list)
    sections_missing: list = field(default_factory=list)
    total_experience_months: Optional[int] = None
    internship_months: Optional[int] = None
    warnings: list = field(default_factory=list)
    extraction_method: str = "text"
    text: str = ""
    content_text: str = ""  # text without header/contact lines; the only text used for scoring

    @property
    def parse_quality(self) -> float:
        """0-1 heuristic of how much of the document was understood (not a measure of resume quality)."""
        q = 0.15 * bool(self.email or self.phone) + 0.10 * bool(self.name)
        found = [s for s in EXPECTED_SECTIONS if s in self.sections_found and s not in self.sections_missing]
        q += 0.35 * len(found) / len(EXPECTED_SECTIONS)
        entries = self.experience + self.education + self.projects
        if entries:
            q += 0.4 * sum(e.confidence in ("high", "medium") for e in entries) / len(entries)
        elif self.skills:
            q += 0.1
        if self.extraction_method in ("pdf-ocr", "pdf-text+ocr"):
            q *= 0.8
        return round(min(q, 1.0), 2)

    def to_dict(self, include_text: bool = False) -> dict:
        d = asdict(self)
        if not include_text:
            d.pop("text", None)
        return d

    @property
    def skill_names(self) -> set:
        return {s.name for s in self.skills}

    def scoring_view(self) -> dict:
        """Job-relevant content only: no name, contact details, links or education dates."""
        return {
            "skills": sorted(self.skill_names),
            "experience": [{"title": e.title, "organization_text": " ".join(e.highlights), "highlights": e.highlights, "months": e.dates.months, "is_internship": e.is_internship} for e in self.experience],
            "projects": [{"name": p.name, "description": p.description, "technologies": p.technologies} for p in self.projects],
            "education": [{"degree": e.degree, "level": e.level, "field": e.field_of_study} for e in self.education],
            "certifications": [{"name": c.name, "issuer": c.issuer} for c in self.certifications],
        }


# ----------------------------------------------------------------------------- helpers
def _norm_heading(line: str) -> str:
    s = BULLET_ONLY_RE.sub("", line).strip().lower().replace("&", " and ").replace("/", " and ")
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_INLINE_HEADINGS = {"summary", "experience", "education", "skills", "projects", "certifications"}


def detect_heading(line: str) -> tuple[Optional[str], str]:
    """Return (section, trailing_content) if ``line`` is a section heading."""
    raw = line.strip()
    if not raw or len(raw) > 60:
        return None, ""
    if ":" in raw:
        head, _, tail = raw.partition(":")
        sec = _HEADING_LOOKUP.get(_norm_heading(head))
        # "Languages: Python, SQL" inside a skills block is a skill category, not a section
        if sec and len(head) < 40 and sec in _INLINE_HEADINGS:
            return sec, tail.strip()
    if BULLET_RE.match(raw) and not raw.isupper():
        return None, ""
    if raw.endswith((".", ",")) and not raw.isupper():
        return None, ""
    sec = _HEADING_LOOKUP.get(_norm_heading(raw))
    return (sec, "") if sec else (None, "")


def split_sections(lines: list[str]) -> tuple[list[str], dict]:
    header, sections, current = [], {}, None
    for ln in lines:
        sec, tail = detect_heading(ln)
        if sec:
            current = sec
            sections.setdefault(sec, [])
            if tail:
                sections[sec].append(tail)
            continue
        (header if current is None else sections[current]).append(ln)
    return header, sections


def _ym(tok: str, end: bool, today: date) -> Optional[str]:
    t = tok.lower().strip()
    if re.match(r"(present|current|now|ongoing|till|to\s+date)", t):
        return "present"
    m = re.match(rf"({_MONTH})\.?,?\s+({_YEAR})", t)
    if m:
        return f"{m.group(2)}-{_MONTHS[m.group(1)[:3]]:02d}"
    m = re.match(rf"(\d{{1,2}})/({_YEAR})", t)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    m = re.match(rf"({_YEAR})", t)
    if m:
        return f"{m.group(1)}-{12 if end else 1:02d}"
    return None


def _to_idx(ym: str, today: date) -> int:
    if ym == "present":
        return today.year * 12 + today.month
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def parse_dates(text: str, today: date) -> tuple[DateSpan, str]:
    """Find a date range (or single date) in ``text``; return the span and the text with it removed."""
    m = DATE_RANGE_RE.search(text)
    if m:
        s, e = _ym(m.group("s"), False, today), _ym(m.group("e"), True, today)
        span = DateSpan(raw=m.group(0), start=s, end=e)
        if s and e:
            n = _to_idx(e, today) - _to_idx(s, today) + 1
            span.months = n if n > 0 else None
            if n <= 0:
                span.start = span.end = None
        return span, (text[: m.start()] + " " + text[m.end():]).strip()
    m = SINGLE_DATE_RE.search(text)
    if m and re.search(_YEAR, m.group(0)):
        return DateSpan(raw=m.group(0), end=_ym(m.group(0), True, today)), (text[: m.start()] + " " + text[m.end():]).strip()
    return DateSpan(), text


def _union_months(spans: list[tuple[int, int]]) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e + 1:
            if cur_e is not None:
                total += cur_e - cur_s + 1
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s + 1
    return total


def _strip_bullet(line: str) -> str:
    return BULLET_RE.sub("", line, count=1).strip()


def _split_parts(text: str) -> list[str]:
    parts = re.split(r"\s*[|•·@]\s*|\s+[–—-]\s+|\t+|\s{3,}|,\s+(?=[A-Z0-9])|\s+at\s+|\s*\(\s*|\s*\)\s*", text)
    return [p.strip(" -–—,:;|()") for p in parts if p and p.strip(" -–—,:;|()")]


# ----------------------------------------------------------------------------- entry segmentation
def _segment_entries(lines: list[str], today: date, loose_headers: bool = False) -> list[dict]:
    """Group section lines into entries: {"header": [...], "body": [...]}."""
    entries, cur, phase, pending_break, last_len = [], None, "header", False, 0

    def push():
        nonlocal cur
        if cur and (cur["header"] or cur["body"]):
            entries.append(cur)
        cur = {"header": [], "body": []}

    push()
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if cur["body"]:
                pending_break = True
            continue
        is_bullet = bool(BULLET_RE.match(line))
        text = _strip_bullet(line) if is_bullet else line.strip()
        if is_bullet:
            phase = "body"
            cur["body"].append(text)
            last_len = len(line)
            pending_break = False
            continue
        if phase == "header":
            if cur["header"] and DATE_RANGE_RE.search(text) and any(DATE_RANGE_RE.search(h) for h in cur["header"]):
                push()  # a second dated header: new entry (layouts without bullets)
                cur["header"].append(text)
            elif cur["header"] and (text.endswith(".") or len(text.split()) >= 9):
                cur["body"].append(text)  # prose after the header line(s) is description, not header
                phase = "body"
            elif len(cur["header"]) < 3 and (len(text) < 140):
                cur["header"].append(text)
            else:
                cur["body"].append(text)
                phase = "body"
            continue
        # body phase, non-bullet line
        has_date = bool(DATE_RANGE_RE.search(text))
        prev = cur["body"][-1] if cur["body"] else ""
        role_like = bool(TITLE_RE.search(text)) and len(text.split()) <= 14 and not text.endswith((".", "!", "?")) and bool(re.search(r"[,|–—@]|\s-\s|\sat\s", text)) and prev.endswith((".", "!", "?", ";"))
        headerish = has_date or role_like or bool(re.search(r"\s[|–—]\s", text)) or pending_break
        wrapped = (text[:1].islower() or (last_len >= 60 and not prev.endswith((".", "!", "?", ";", ":")))) and not headerish
        if wrapped and cur["body"]:
            cur["body"][-1] += " " + text
            last_len = len(text) + 2
        elif headerish or (not _TECH_LINE_RE.match(text) and len(text) < 60 and prev.endswith((".", "!", "?")) and (loose_headers and len(text.split()) <= 8 and text[:1].isupper() and ":" not in text and not text.endswith((".", "!", "?")) or TITLE_RE.search(text) or ORG_RE.search(text))):
            push()
            phase = "header"
            cur["header"].append(text)
            pending_break = False
        else:
            cur["body"].append(text)
            last_len = len(text) + 2
    push()
    return entries


# ----------------------------------------------------------------------------- field parsers
def _parse_experience(lines: list[str], today: date) -> list[ExperienceEntry]:
    out = []
    for e in _segment_entries(lines, today):
        header = " | ".join(e["header"])
        if not header and not e["body"]:
            continue
        span, rest = parse_dates(header, today)
        parts = _split_parts(rest)
        title = org = None
        for p in parts:
            if title is None and TITLE_RE.search(p) and len(p.split()) <= 8:
                title = p
            elif org is None and (ORG_RE.search(p) or not TITLE_RE.search(p)) and not re.fullmatch(r"[\d\W]+", p) and len(p) <= 70:
                org = p
        if title and not org and len(parts) > 1:
            others = [p for p in parts if p != title]
            org = others[0] if others else None
        score = sum(bool(x) for x in (title, org, span.start or span.raw))
        conf = "high" if score == 3 else "medium" if score == 2 else "low"
        text_blob = " ".join(e["body"])
        sks = sorted({c for c, _, _ in sk.find_skills(text_blob + " " + header)})
        out.append(ExperienceEntry(title=title, organization=org, dates=span, highlights=e["body"], skills=sks,
                                   is_internship=bool(re.search(r"\bintern(ship)?\b|\btrainee\b", header, re.I)), confidence=conf,
                                   raw="\n".join(e["header"] + e["body"])))
    # free-text lines (e.g. "Fresher: coursework listed below") are not jobs
    def _sentence(o):  # a free-text sentence, not an employer name
        return bool(o) and (len(o.split()) > 6 or (o.rstrip().endswith((".", ";", "!")) and not ORG_RE.search(o)))
    out = [x for x in out if not (not x.title and not x.dates.raw and _sentence(x.organization))]
    return [x for x in out if x.title or x.organization or x.dates.raw]


def _degree_in(text: str) -> Optional[tuple[str, int, int, int]]:
    """Return (label_text, level, start, end) of the earliest degree mention."""
    best = None
    for rx, lvl, _ in _DEGREE_RES:
        m = rx.search(text)
        if m and (best is None or m.start() < best[2]):
            best = (m.group(0), lvl, m.start(), m.end())
    for m in _ABBREV_RE.finditer(text):
        lvl = _ABBREV_LEVEL.get(m.group(1)[0].upper())
        if lvl and (best is None or m.start() < best[2]):
            best = (m.group(0), lvl, m.start(), m.end())
            break
    return best


def _is_role_line(t: str) -> bool:
    t = t.strip()
    return bool(t) and not BULLET_RE.match(t) and bool(TITLE_RE.search(t)) and len(t.split()) <= 14 and not t.endswith((".", "!", "?")) and \
        bool(re.search(r"[,|–—@]|\s-\s|\sat\s", t)) and (bool(DATE_RANGE_RE.search(t)) or bool(ORG_RE.search(t)) or "|" in t)


def _infer_experience_lines(lines: list[str], sections: dict) -> list[str]:
    """When no experience heading exists, take the block that starts at the first job-title line and runs until
    education / skills / a recognised heading. Used only as a low-confidence fallback."""
    start = next((i for i, l in enumerate(lines) if detect_heading(l)[0] is None and _is_role_line(l)), None)
    if start is None:
        return []
    out = []
    for l in lines[start:]:
        sec, _ = detect_heading(l)
        if sec and sec != "experience":
            break
        if re.match(r"^\s*skills?\s*:", l, re.I) or (_degree_in(l) and INSTITUTION_RE.search(l)):
            break
        out.append(l)
    return out


def _unwrap_education(lines: list[str]) -> list[str]:
    """Re-join entries that a narrow column or page width wrapped mid-name ("Fairhaven" / "Institute, 2019-2022")."""
    out = []
    for ln in lines:
        t = ln.strip()
        first = t.split()[0].lower().strip(",.") if t.split() else ""
        second = t.split()[1].lower() if len(t.split()) > 1 else ""
        starts_with_inst_word = bool(INSTITUTION_RE.fullmatch(first)) and second not in ("of", "for", "in", "at") and not BULLET_RE.match(t)
        prev_open = bool(out) and out[-1].rstrip().endswith((",", "&", "-", "–", "—", " of", " and", " for", " in"))
        if out and t and (starts_with_inst_word or prev_open or t[:1].islower()):
            out[-1] = out[-1].rstrip() + " " + t
        else:
            out.append(ln)
    return out


def _parse_education(lines: list[str], today: date) -> list[EducationEntry]:
    lines = _unwrap_education(lines)
    groups, cur = [], []

    def flush():
        nonlocal cur
        if cur:
            groups.append(cur)
        cur = []

    def has_deg(ls):
        return any(_degree_in(l) for l in ls)

    def has_inst(ls):
        return any(INSTITUTION_RE.search(l) for l in ls)

    for ln in lines:
        if not ln.strip():
            if cur and has_deg(cur) and has_inst(cur):
                flush()
            continue
        t = _strip_bullet(ln) if BULLET_RE.match(ln) else ln.strip()
        if cur and ((_degree_in(t) and has_deg(cur)) or (INSTITUTION_RE.search(t) and has_inst(cur) and has_deg(cur))):
            flush()
        cur.append(t)
    flush()
    out = []
    for g in groups:
        blob = " | ".join(g)
        span, rest = parse_dates(blob, today)
        gm = GPA_RE.search(rest)
        grade = (gm.group(1) or gm.group(2)).strip() if gm else None
        if gm:
            rest = rest.replace(gm.group(0), " ")
        deg = _degree_in(rest)
        degree = level = fld = inst = None
        if deg:
            label, level, s, e = deg
            tail = rest[e:]
            fm = re.match(r"\s*(?:\(|,|:|-|–|\bin\b|\bof\b|\bspecializing in\b)\s*(?:in\s+|of\s+)?([A-Za-z&/ \-]{3,60})", tail)
            if fm:
                fld = re.split(r"\s*[|,(]\s*|\s+(?:from|at|university|college|institute)\b", fm.group(1), flags=re.I)[0].strip(" -") or None
            degree = label.strip()
            m2 = re.match(r"(master|bachelor)(?:'?s)?\s+of\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", rest[s:], re.I)
            if m2 and re.search(r"\bin\b", rest[s:]):
                degree = rest[s: s + m2.end()].strip()
        for p in _split_parts(rest):
            if INSTITUTION_RE.search(p) and not (deg and p.strip() == rest[deg[2]:deg[3]].strip()):
                inst = p
                break
        if inst and deg and deg[2] < rest.find(inst) < deg[3] + 80 and fld and inst.startswith(fld):
            fld = None
        if deg or inst:
            conf = "high" if deg and inst and (span.raw or True) and fld else "medium" if deg and inst else "low"
            out.append(EducationEntry(degree=degree, level=level, field_of_study=fld, institution=inst, dates=span, grade=grade, confidence=conf, raw="\n".join(g)))
    return out


_TECH_LINE_RE = re.compile(r"^\s*(?:technologies|tech(?:nologies)?(?: used)?|tech stack|stack|built with|tools(?: used)?|technologies used|libraries|frameworks|environment)\s*[:\-–]\s*(.+)$", re.I)


def _parse_projects(lines: list[str], today: date) -> list[ProjectEntry]:
    out = []
    for e in _segment_entries(lines, today, loose_headers=True):
        if not e["header"]:
            continue
        head = e["header"][0]
        extra_head = e["header"][1:]
        urls = [u.rstrip(".") for u in URL_RE.findall(" ".join(e["header"] + e["body"]))]
        name_raw = URL_RE.sub("", head)
        parts = [p for p in re.split(r"\s+[|–—]\s+|\s*\|\s*|:\s+", name_raw) if p.strip()]
        name = parts[0].strip(" -–—:|") if parts else None
        techs = set()
        for p in parts[1:]:
            techs |= {c for c, _, _ in sk.find_skills(p)}
        desc, achievements = [], []
        for ln in extra_head + e["body"]:
            tm = _TECH_LINE_RE.match(ln)
            if tm:
                techs |= {c for c, _, _ in sk.find_skills(tm.group(1))}
                continue
            desc.append(ln)
            if RESULT_RE.search(ln):
                achievements.append(ln)
        techs |= {c for c, _, _ in sk.find_skills(" ".join(desc))}
        if name and len(name) > 90:
            desc.insert(0, name)
            name = name[:80].rsplit(" ", 1)[0] + "…"
        has = sum([bool(name), bool(desc), bool(techs)])
        out.append(ProjectEntry(name=name, description=desc, technologies=sorted(techs), achievements=achievements, links=urls,
                                confidence="high" if has == 3 else "medium" if has == 2 else "low", raw="\n".join(e["header"] + e["body"])))
    return out


def _parse_certs(lines: list[str]) -> list[CertificationEntry]:
    items = []
    for ln in lines:
        if not ln.strip():
            continue
        t = _strip_bullet(ln) if BULLET_RE.match(ln) else ln.strip()
        if items and t[:1].islower() and not BULLET_RE.match(ln):
            items[-1] += " " + t
        else:
            items.append(t)
    out = []
    for it in items:
        year = None
        ym = re.search(_YEAR, it)
        if ym:
            year = ym.group(0)
        clean = re.sub(rf"\(?\b(?:{_MONTH}\.?\s+)?{_YEAR}\b\)?", "", it, flags=re.I).strip(" -–—,|:")
        issuer, name, conf = None, clean, "low"
        m = re.match(r"(.+?)\s*(?:\bissued by\b|\bby\b|\bfrom\b)\s+(.+)$", clean, re.I)
        parts = [p for p in re.split(r"\s+[–—|-]\s+|\s*\|\s*|,\s+", clean) if p.strip()]
        known = [p for p in parts if any(re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", p.lower()) for k in KNOWN_ISSUERS)]
        if m:
            name, issuer, conf = m.group(1).strip(), m.group(2).strip(" ."), "high"
        elif len(parts) >= 2:
            if known and known[-1] != parts[0]:
                issuer = known[-1]
                name = " - ".join(p for p in parts if p != issuer)
                conf = "high"
            elif known and known[0] == parts[0]:
                issuer, name, conf = parts[0], " - ".join(parts[1:]), "medium"
            else:
                name, issuer, conf = parts[0], parts[1], "low"
        elif known:
            conf = "medium"
        elif CERT_WORD_RE.search(clean):
            conf = "medium"
        if len(name) < 3 or len(name) > 150:
            continue
        out.append(CertificationEntry(name=name.strip(), issuer=issuer, year=year, confidence=conf))
    return out


def _parse_skill_section(lines: list[str]) -> tuple[set, list]:
    listed, other = set(), []
    for ln in lines:
        t = _strip_bullet(ln) if BULLET_RE.match(ln) else ln.strip()
        if not t:
            continue
        if ":" in t and len(t.split(":")[0]) < 40:
            t = t.split(":", 1)[1]
        for c, _, _ in sk.find_skills(t):
            listed.add(c)
        for item in re.split(r"\s*[,;|•·]\s*|\s{3,}", t):
            item = item.strip(" .()-")
            if not item or len(item) > 40 or len(item.split()) > 4:
                continue
            if not sk.skill_set(item) and re.search(r"[A-Za-z]", item) and item.lower() not in {"and", "etc", "others", "more"}:
                other.append(item)
    return listed, list(dict.fromkeys(other))


# ----------------------------------------------------------------------------- contact
def _find_name(header_lines: list[str], email: Optional[str]) -> tuple[Optional[str], str]:
    for i, ln in enumerate(header_lines[:6]):
        t = ln.strip()
        if not t or "@" in t or URL_RE.search(t) or re.search(r"\d{3}", t) or len(t) > 50:
            continue
        t2 = re.sub(r"\b(resume|curriculum vitae|cv)\b", "", t, flags=re.I).strip(" -–|,")
        words = t2.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'.\-]*", w) for w in words) and not TITLE_RE.search(t2) and not _HEADING_LOOKUP.get(_norm_heading(t2)):
            if t2.isupper() or all(w[0].isupper() for w in words):
                return t2.title() if t2.isupper() else t2, "high" if i == 0 else "medium"
    if email:  # last resort: derive from the e-mail local part (flagged low confidence)
        local = re.split(r"[._\-]", email.split("@")[0])
        local = [p for p in local if p.isalpha() and len(p) > 1]
        if len(local) >= 2:
            return " ".join(p.capitalize() for p in local[:3]), "low"
    return None, "low"


def _find_phone(text: str) -> Optional[str]:
    for m in PHONE_RE.finditer(text):
        cand = m.group(0).strip()
        digits = re.sub(r"\D", "", cand)
        if 10 <= len(digits) <= 13 and not re.fullmatch(rf"{_YEAR}\s*[-–]\s*{_YEAR}", cand) and not re.search(_YEAR + r"\s*[-–]\s*" + _YEAR, cand):
            return cand
    return None


def _classify_links(urls: list[str]) -> dict:
    out = {"other": []}
    for u in urls:
        u = u.rstrip(".,;)")
        low = u.lower()
        if u.startswith("mailto:") or "@" in u and not low.startswith("http"):
            continue
        if "linkedin.com" in low and "linkedin" not in out:
            out["linkedin"] = u
        elif "github.com" in low and "github" not in out and low.rstrip("/").count("/") <= 4 and "github" not in out:
            out["github"] = u
        elif u not in out["other"]:
            out["other"].append(u)
    if not out["other"]:
        out.pop("other")
    return out


# ----------------------------------------------------------------------------- public API
EXPECTED_SECTIONS = ["experience", "education", "skills", "projects"]


def parse_resume(text: str, extra_links: Optional[list] = None, extraction_method: str = "text", today: Optional[date] = None) -> ParsedResume:
    today = today or date.today()
    pr = ParsedResume(text=text, extraction_method=extraction_method)
    lines = text.split("\n")
    header, sections = split_sections(lines)
    # Scoring text excludes sections that are not job-relevant evidence or can act as demographic proxies
    # (institution names, education dates, hobbies, references); education is re-added as degree + field below.
    _excluded = {"education", "interests", "references", "languages"}
    body_lines = [l for name, sec in sections.items() if name not in _excluded for l in sec] if sections else lines
    scrub = lambda t: URL_RE.sub(" ", EMAIL_RE.sub(" ", t))
    # dates are not content: strip them so tenure wording/years cannot leak into lexical similarity
    pr.content_text = SINGLE_DATE_RE.sub(" ", DATE_RANGE_RE.sub(" ", scrub("\n".join(body_lines))))
    pr.sections_found = list(sections)
    pr.sections_missing = [s for s in EXPECTED_SECTIONS if s not in sections or not any(l.strip() for l in sections[s])]

    # contact
    em = EMAIL_RE.search(text)
    pr.email = em.group(0) if em else None
    head_text = "\n".join(header) if header else "\n".join(lines[:8])
    pr.phone = _find_phone(head_text) or _find_phone("\n".join(lines[:25]))
    urls = [u for u in URL_RE.findall(text)] + list(extra_links or [])
    pr.links = _classify_links([u for u in urls if not (em and pr.email and u.lower() == pr.email.lower())])
    pr.name, pr.name_confidence = _find_name(header or lines[:6], pr.email)

    # summary
    if "summary" in sections:
        pr.summary = " ".join(l.strip() for l in sections["summary"] if l.strip())[:1200]
    elif header:
        prose = [l.strip() for l in header[1:] if len(l.split()) >= 12]
        pr.summary = " ".join(prose)[:1200]

    # structured sections
    if "experience" in sections:
        pr.experience = _parse_experience(sections["experience"], today)
    if not pr.experience:
        inferred = _infer_experience_lines(lines, sections)
        if inferred:
            pr.experience = [e for e in _parse_experience(inferred, today) if e.title]  # fallback keeps only entries with a job title
            for e in pr.experience:
                e.confidence = "low"
            pr.warnings.append("No experience heading was recognised; roles were inferred from job-title lines (low confidence).")
    if "education" in sections:
        pr.education = _parse_education(sections["education"], today)
    if "projects" in sections:
        pr.projects = _parse_projects(sections["projects"], today)
    if "certifications" in sections:
        pr.certifications = _parse_certs(sections["certifications"])
    else:
        loose = [l for l in lines if re.search(r"\bcertified\b|\bcertification\b", l, re.I) and len(l) < 120 and not BULLET_RE.match(l)]
        if loose:
            pr.certifications = [c for c in _parse_certs(loose)]
            for c in pr.certifications:
                c.confidence = "low"
            pr.warnings.append("No certifications section found; certification-like lines were picked up from other sections (low confidence).")
    for key, attr in (("awards", "awards"), ("languages", "languages"), ("publications", "publications")):
        if key in sections:
            setattr(pr, attr, [_strip_bullet(l) if BULLET_RE.match(l) else l.strip() for l in sections[key] if l.strip()])

    # a bare education line without section (e.g. header-only resume)
    if not pr.education and "education" not in sections:
        deg_lines = [l for l in lines if _degree_in(l) and INSTITUTION_RE.search(l)]
        if deg_lines:
            pr.education = _parse_education(deg_lines, today)
            for e in pr.education:
                e.confidence = "low"
            pr.warnings.append("No education section heading found; education was inferred from degree lines (low confidence).")

    pr.content_text += "\n" + "\n".join(" ".join(x for x in (e.degree, "in " + e.field_of_study if e.field_of_study else None) if x) for e in pr.education)

    # skills (taxonomy-normalised) with evidence
    listed, other = _parse_skill_section(sections.get("skills", []))
    pr.other_listed_skills = other
    recs: dict[str, SkillRecord] = {}

    def add(canon, section, surface, line):
        r = recs.setdefault(canon, SkillRecord(canon, sk.CATEGORY.get(canon, "Other"), canon in listed))
        if section not in r.sections:
            r.sections.append(section)
        if surface not in r.surface_forms:
            r.surface_forms.append(surface)
        if line and section in ("experience", "projects") and len(r.evidence) < 3:
            r.evidence.append({"section": section, "text": line.strip()[:220]})

    for sec, sec_lines in (sections.items() if sections else [("document", lines)]):
        if sec in ("references", "interests", "languages"):
            continue
        for ln in sec_lines:
            for canon, surf, _ in sk.find_skills(ln):
                add(canon, sec, surf, ln)
    if header:
        for ln in header:
            for canon, surf, _ in sk.find_skills(ln):
                add(canon, "summary", surf, None)
    # project technology lists count as project-section evidence
    for p in pr.projects:
        for t in p.technologies:
            if t in recs and "projects" not in recs[t].sections:
                recs[t].sections.append("projects")
    pr.skills = sorted(recs.values(), key=lambda r: (not r.listed, r.category, r.name))

    # experience duration
    spans, intern_spans = [], []
    for e in pr.experience:
        d = e.dates
        if d.start and d.end:
            a, b = _to_idx(d.start, today), _to_idx(d.end, today)
            if b >= a:
                (intern_spans if e.is_internship else spans).append((a, b))
    if spans or intern_spans:
        pr.total_experience_months = _union_months(spans)
        pr.internship_months = _union_months(intern_spans)

    for pii in (pr.name, pr.phone):  # when no sections were found the scoring text still contains the header
        if pii:
            pr.content_text = pr.content_text.replace(pii, " ")

    # warnings
    if not sections:
        pr.warnings.append("No recognised section headings were found; structured fields (experience, education, projects) could not be separated and may be incomplete.")
    for s in pr.sections_missing:
        if sections:
            pr.warnings.append(f"No '{s}' section detected.")
    if "experience" in sections and not pr.experience:
        pr.warnings.append("An experience section exists but no entries could be parsed.")
    if pr.experience and not any(e.dates.raw for e in pr.experience):
        pr.warnings.append("No dates were detected in experience entries; total experience cannot be estimated.")
    if not pr.email:
        pr.warnings.append("No e-mail address detected.")
    low = [e for e in pr.experience if e.confidence == "low"]
    if low:
        pr.warnings.append(f"{len(low)} experience entr{'y' if len(low) == 1 else 'ies'} parsed with low confidence (title/company/dates unclear).")
    if extraction_method in ("pdf-ocr", "pdf-text+ocr"):
        pr.warnings.append("Text came partly or fully from OCR; all fields are lower confidence.")
    return pr
