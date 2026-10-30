"""Document -> text. Supports PDF (text layer, OCR fallback for scans) and DOCX."""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field

MAX_BYTES = 20 * 1024 * 1024
MIN_CHARS_PER_PAGE = 40  # below this a page is treated as image-only
OCR_DPI = 200
MAX_OCR_PAGES = 6


class ExtractionError(Exception):
    """Raised when a document cannot be turned into usable text."""


@dataclass
class ExtractionResult:
    text: str
    method: str  # "pdf-text" | "pdf-ocr" | "pdf-text+ocr" | "docx" | "txt"
    pages: int = 1
    warnings: list = field(default_factory=list)
    ocr_used: bool = False
    links: list = field(default_factory=list)  # hyperlink targets stored behind link text


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = text.replace("•", "•").replace("", "•").replace("▪", "•").replace("", "•")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_available() -> bool:
    """True if Tesseract OCR can be used. Set RESUME_INTEL_DISABLE_OCR=1 to force-disable (used in tests)."""
    import os

    if os.environ.get("RESUME_INTEL_DISABLE_OCR") == "1":
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page(page) -> str:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _find_gutter(page):
    """Detect a clean two-column layout. Returns (gutter_x, header_bottom) or None.

    A gutter is a vertical band no text block crosses, with substantial text on both sides. Blocks that cross it must all
    lie above the columns (a full-width header); otherwise the page is treated as single-column.
    """
    W = page.rect.width
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if len(blocks) < 6:
        return None
    best = None
    steps = [W * (0.25 + 0.01 * i) for i in range(0, 51)]
    for gx in steps:
        crossing = [b for b in blocks if b[0] < gx < b[2]]
        left = [b for b in blocks if b[2] <= gx]
        right = [b for b in blocks if b[0] >= gx]
        if len(left) < 3 or len(right) < 3 or min(len(left), len(right)) < 0.15 * len(blocks):
            continue
        col_top = min(min(b[1] for b in left), min(b[1] for b in right))
        if any(b[3] > col_top + 2 for b in crossing):
            continue  # something spans the gutter inside the columns -> not a simple two-column page
        gap = min(b[0] for b in right) - max(b[2] for b in left)
        if gap >= 8 and (best is None or gap > best[2]):
            best = (gx, max([b[3] for b in crossing], default=0.0) + 1, gap)
    return (best[0], best[1]) if best else None


def _page_text(page) -> str:
    g = _find_gutter(page)
    if g is None:
        return page.get_text("text", sort=True)
    import fitz

    gx, top = g
    r = page.rect
    parts = []
    if top > 1:
        parts.append(page.get_text("text", clip=fitz.Rect(r.x0, r.y0, r.x1, top), sort=True))
    parts.append(page.get_text("text", clip=fitz.Rect(r.x0, top, gx, r.y1), sort=True))
    parts.append(page.get_text("text", clip=fitz.Rect(gx, top, r.x1, r.y1), sort=True))
    return "\n".join(parts)


def _extract_pdf(data: bytes) -> ExtractionResult:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("PyMuPDF is required to read PDF files") from exc
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF (corrupt or not a PDF): {exc}") from exc
    warnings: list = []
    links: list = []
    with doc:
        if doc.needs_pass:
            raise ExtractionError("PDF is password-protected")
        if doc.page_count == 0:
            raise ExtractionError("PDF has no pages")
        texts, ocr_pages = [], 0
        ocr_ok = None
        for i, page in enumerate(doc):
            try:
                t = _page_text(page)
            except Exception as exc:
                warnings.append(f"Page {i + 1}: text layer unreadable ({exc})")
                t = ""
            if len(t.strip()) < MIN_CHARS_PER_PAGE and page.get_images():
                if ocr_ok is None:
                    ocr_ok = ocr_available()
                    if not ocr_ok:
                        warnings.append("Image-only pages detected but OCR (Tesseract) is not installed; text from those pages is missing.")
                if ocr_ok and ocr_pages < MAX_OCR_PAGES:
                    try:
                        t = _ocr_page(page)
                        ocr_pages += 1
                    except Exception as exc:
                        warnings.append(f"Page {i + 1}: OCR failed ({exc})")
            texts.append(t)
            try:
                links.extend(l["uri"] for l in page.get_links() if l.get("uri"))
            except Exception:
                pass
        n = doc.page_count
    text = _clean("\n".join(texts))
    if ocr_pages:
        warnings.append(f"OCR was used on {ocr_pages} page(s); recognised text may contain errors, so treat extracted details as lower confidence.")
    method = "pdf-text" if not ocr_pages else ("pdf-ocr" if ocr_pages == n else "pdf-text+ocr")
    return ExtractionResult(text, method, n, warnings, ocr_used=bool(ocr_pages), links=list(dict.fromkeys(links)))


def _extract_docx(data: bytes) -> ExtractionResult:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("python-docx is required to read DOCX files") from exc
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise ExtractionError("File is not a valid DOCX (not a zip container)")
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not open DOCX: {exc}") from exc
    lines = []
    for sec in d.sections:
        for p in sec.header.paragraphs:
            if p.text.strip():
                lines.append(p.text)
    # body paragraphs and tables in document order
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in d.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, d)
            txt = p.text
            if txt.strip():
                is_list = p.style is not None and "list" in (p.style.name or "").lower()
                lines.append(("• " + txt) if is_list and not txt.lstrip().startswith(("•", "-", "*")) else txt)
            else:
                lines.append("")
        elif child.tag.endswith("}tbl"):
            for row in Table(child, d).rows:
                seen, cells = set(), []
                for c in row.cells:
                    if id(c._tc) in seen:
                        continue
                    seen.add(id(c._tc))
                    if c.text.strip():
                        cells.append(c.text.strip().replace("\n", " "))
                if cells:
                    lines.append("  |  ".join(cells))
    links = [r.target_ref for r in d.part.rels.values() if r.reltype.endswith("/hyperlink") and r.is_external]
    return ExtractionResult(_clean("\n".join(lines)), "docx", 1, [], links=list(dict.fromkeys(links)))


def extract_text(data: bytes, filename: str = "") -> ExtractionResult:
    """Extract text from PDF/DOCX bytes. Raises ExtractionError with a user-readable reason."""
    if not data:
        raise ExtractionError("File is empty")
    if len(data) > MAX_BYTES:
        raise ExtractionError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB")
    name = filename.lower()
    if data[:5] == b"%PDF-" or name.endswith(".pdf"):
        res = _extract_pdf(data)
    elif data[:2] == b"PK" or name.endswith(".docx"):
        res = _extract_docx(data)
    elif name.endswith(".txt"):
        res = ExtractionResult(_clean(data.decode("utf-8", errors="replace")), "txt")
    elif name.endswith(".doc"):
        raise ExtractionError("Legacy .doc files are not supported; save the resume as .docx or .pdf")
    else:
        raise ExtractionError("Unsupported file type (supported: PDF, DOCX)")
    if len(res.text) < 30:
        hint = "" if res.ocr_used or not res.method.startswith("pdf") else " If this is a scanned PDF, install Tesseract OCR to enable text recognition."
        raise ExtractionError("Almost no text could be extracted." + hint)
    return res
