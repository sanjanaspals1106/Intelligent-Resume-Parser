"""End-to-end orchestration: file -> parsed resume -> match -> assessment, plus batch ranking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Optional

from .assessment import Assessment, assess
from .extraction import ExtractionError, extract_text
from .jobs import JobDescription, parse_job_description
from .matching import MatchResult, match
from .parser import ParsedResume, parse_resume
from .semantic import Encoder, get_encoder

NEAR_TIE_POINTS = 3.0


@dataclass
class Analysis:
    filename: str
    status: str  # "ok" | "error"
    error: Optional[str] = None
    resume: Optional[ParsedResume] = None
    match: Optional[MatchResult] = None
    assessment: Optional[Assessment] = None
    seconds: float = 0.0
    extraction_warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self, include_text: bool = False, anonymize_as: Optional[str] = None) -> dict:
        """``anonymize_as="Candidate 3"`` replaces the file name and strips name, e-mail, phone, links and raw text."""
        resume = self.resume.to_dict(include_text and not anonymize_as) if self.resume else None
        if resume and anonymize_as:
            resume.update(name=None, email=None, phone=None, links={}, text=None)
            resume.pop("content_text", None)
        return {
            "filename": anonymize_as or self.filename, "status": self.status, "error": self.error, "seconds": round(self.seconds, 3),
            "extraction_warnings": self.extraction_warnings, "resume": resume,
            "match": self.match.to_dict() if self.match else None,
            "assessment": self.assessment.to_dict() if self.assessment else None,
        }


def analyze_text(text: str, jd: JobDescription, filename: str = "resume", encoder: Optional[Encoder] = None, extra_links=None,
                 method: str = "text", today: Optional[date] = None, warnings: Optional[list] = None) -> Analysis:
    t0 = time.perf_counter()
    try:
        resume = parse_resume(text, extra_links=extra_links, extraction_method=method, today=today)
        m = match(resume, jd, encoder=encoder)
        a = assess(resume, jd, m)
        return Analysis(filename, "ok", None, resume, m, a, time.perf_counter() - t0, list(warnings or []))
    except Exception as exc:  # never let one document break a batch
        return Analysis(filename, "error", f"Analysis failed ({type(exc).__name__}); the file may use an unsupported layout.", seconds=time.perf_counter() - t0)


def analyze_file(data: bytes, filename: str, jd: JobDescription, encoder: Optional[Encoder] = None, today: Optional[date] = None) -> Analysis:
    t0 = time.perf_counter()
    try:
        ex = extract_text(data, filename)
    except ExtractionError as exc:
        return Analysis(filename, "error", str(exc), seconds=time.perf_counter() - t0)
    except Exception as exc:
        return Analysis(filename, "error", f"Unexpected extraction error: {type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0)
    res = analyze_text(ex.text, jd, filename, encoder, ex.links, ex.method, today, ex.warnings)
    res.seconds = time.perf_counter() - t0
    return res


def analyze_batch(files: Iterable[tuple], jd_text: str, encoder: Optional[Encoder] = None, today: Optional[date] = None,
                  progress: Optional[Callable[[int, int, str], None]] = None) -> list[Analysis]:
    jd = parse_job_description(jd_text)
    enc = encoder or get_encoder()
    files = list(files)
    out = []
    for i, (name, data) in enumerate(files):
        if progress:
            progress(i, len(files), name)
        out.append(analyze_file(data, name, jd, enc, today))
    if progress:
        progress(len(files), len(files), "done")
    return out


def rank(analyses: list[Analysis], anonymize: bool = False) -> list[dict]:
    """Deterministic ranking: score desc, then required coverage desc, then filename. Failed files are listed last.
    With ``anonymize`` the candidate and file columns carry only "Candidate N" labels (for blind review and exports)."""
    ok = [a for a in analyses if a.ok]
    ok.sort(key=lambda a: (-round(a.match.score, 2), -a.match.required_coverage, a.filename.lower()))
    rows = []
    for i, a in enumerate(ok):
        m, r = a.match, a.resume
        nxt = ok[i + 1].match.score if i + 1 < len(ok) else None
        rows.append({
            "rank": i + 1, "candidate": f"Candidate {i + 1}" if anonymize else (r.name or a.filename), "file": f"candidate_{i + 1}" if anonymize else a.filename, "score": round(m.score, 1), "band": m.band,
            "required_coverage": round(100 * m.required_coverage), "preferred_coverage": round(100 * m.preferred_coverage),
            "matched_required": [s.skill for s in m.matched_required], "missing_required": [s.skill for s in m.missing_required],
            "possibly_missed": [s.skill for s in m.missing_required if s.uncertain],
            "experience_years": round((r.total_experience_months or 0) / 12, 1) if r.total_experience_months is not None else None,
            "baseline_score": round(m.baseline["score"], 1), "confidence": m.confidence, "completeness": a.assessment.completeness_score,
            "near_tie_with_next": nxt is not None and abs(m.score - nxt) < NEAR_TIE_POINTS, "seconds": round(a.seconds, 2),
            "status": "ok", "error": "",
            **{f"dim_{d.key}": (round(100 * d.score) if d.score is not None else None) for d in m.dimensions},
        })
    for n, a in enumerate([x for x in analyses if not x.ok], 1):
        label = f"failed_file_{n}" if anonymize else a.filename
        rows.append({"rank": None, "candidate": label, "file": label, "score": None, "band": "", "status": "error", "error": a.error, "seconds": round(a.seconds, 2)})
    return rows


def anonymized_labels(analyses: list[Analysis]) -> dict:
    """Map filename -> 'Candidate N' using the same ordering as ``rank``."""
    ok = [a for a in analyses if a.ok]
    ok.sort(key=lambda a: (-round(a.match.score, 2), -a.match.required_coverage, a.filename.lower()))
    labels = {a.filename: f"Candidate {i + 1}" for i, a in enumerate(ok)}
    for n, a in enumerate([x for x in analyses if not x.ok], 1):
        labels[a.filename] = f"failed_file_{n}"
    return labels
