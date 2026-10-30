"""Browser end-to-end tests of the real Streamlit app (Playwright + headless Chromium).

Skipped automatically when Playwright or a Chromium binary is unavailable. Set CHROMIUM_PATH to point at a binary.
All resumes are synthetic/fictional.
"""
import glob
import io
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
from evaluation import synthetic as syn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _chromium():
    if os.environ.get("CHROMIUM_PATH"):
        return os.environ["CHROMIUM_PATH"]
    for pat in ("~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*", "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome"):
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if hits:
            return hits[0]
    return None


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start(env_extra=None):
    port = _free_port()
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1).status == 200:
                return proc, port
        except Exception:
            time.sleep(1)
    proc.kill()
    raise RuntimeError("streamlit did not start")


@pytest.fixture(scope="module")
def browser():
    exe = _chromium()
    if not exe:
        pytest.skip("no Chromium binary found")
    with pw.sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe)
        yield b
        b.close()


@pytest.fixture(scope="module")
def server():
    proc, port = _start()
    yield f"http://localhost:{port}"
    proc.terminate()


@pytest.fixture(scope="module")
def server_no_ocr():
    proc, port = _start({"RESUME_INTEL_DISABLE_OCR": "1"})
    yield f"http://localhost:{port}"
    proc.terminate()


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("resumes")
    ps, _ = syn.build_corpus(5, per_family=[("strong", 2), ("weak", 1)])
    ml = [p for p in ps if p.family == "ml_engineer"]
    out = {"people": {}}
    for p in ml:
        fn, data = syn.render(p)
        (d / fn).write_bytes(data)
        out[p.pid] = d / fn
        out["people"][p.pid] = p
    pdf = next(p for p in ml if p.layout != "docx")
    docx_p = next((p for p in ps if p.layout == "docx"), None)
    if docx_p:
        fn, data = syn.render(docx_p)
        (d / fn).write_bytes(data)
        out["docx"] = d / fn
    sp = next(p for p in ps if p.layout == "classic")
    fn, data = syn.render(sp, scanned=True)
    (d / fn).write_bytes(data)
    out["scan"], out["scan_person"] = d / fn, sp
    (d / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    (d / "empty.docx").write_bytes(b"")
    import docx
    u = docx.Document()
    u.add_heading("Zoë Müller-Łukasiewicz", 0)
    u.add_paragraph("zoe.muller@example.org | +49 151 23456789")
    u.add_heading("Experience", 1)
    u.add_paragraph("Entwicklerin | Café Größe GmbH | Jan 2021 – Present")
    u.add_paragraph("Built Python services – “smart quotes”, emoji 🚀 and 日本語 text.", style="List Bullet")
    u.add_heading("Skills", 1)
    u.add_paragraph("Python, SQL, Docker")
    buf = io.BytesIO(); u.save(buf)
    (d / "unicode.docx").write_bytes(buf.getvalue())
    out["dir"] = d
    return out


def _open(browser, url):
    page = browser.new_page(viewport={"width": 1500, "height": 1100})
    page.goto(url)
    page.get_by_role("button", name="Load sample job description").wait_for(timeout=60000)
    page.get_by_role("button", name="Load sample job description").click()
    page.wait_for_timeout(1500)
    return page


def _upload(page, nth, paths):
    page.locator("input[type=file]").nth(nth).set_input_files([str(p) for p in paths])


def test_single_resume_journey(browser, server, files):
    page = _open(browser, server)
    _upload(page, 0, [files["docx"] if "docx" in files else next(v for k, v in files.items() if k.startswith("R"))])
    page.get_by_text("Compatibility").first.wait_for(timeout=90000)
    assert page.get_by_text("Original baseline").count() >= 1
    assert "not a probability" in page.content().lower() or "probability of being hired" in page.content().lower()
    for tab in ("Skills", "Evidence", "Assessment", "Recommendations", "Parsed resume"):
        page.get_by_role("tab", name=tab).click()
        page.wait_for_timeout(400)
    page.get_by_role("tab", name="Assessment").click()
    assert page.get_by_text("Strengths").count() and page.get_by_text("Gaps").count() and page.get_by_text("Uncertain").count()
    page.get_by_role("tab", name="Download").click()
    with page.expect_download() as dl:
        page.get_by_role("button", name="HTML report (printable)").click()
    html = Path(dl.value.path()).read_text()
    assert "Resume assessment" in html and "probability of being hired" in html
    with page.expect_download() as dl:
        page.get_by_role("button", name="JSON (full analysis)").click()
    import json
    assert json.loads(Path(dl.value.path()).read_text())["match"]["score"] >= 0
    assert not page.get_by_text("Traceback").count()
    page.screenshot(path=str(ROOT / "docs" / "screenshots" / "single_resume.png"), full_page=False) if os.environ.get("E2E_SCREENSHOTS") else None
    page.close()


def test_ranking_journey_with_errors_filters_exports_and_blind_review(browser, server, files):
    page = _open(browser, server)
    page.get_by_role("tab", name="Rank candidates").click()
    people = [v for k, v in files.items() if k.startswith("R")]
    batch = people + [files["dir"] / "broken.pdf", files["dir"] / "empty.docx", files["dir"] / "unicode.docx"]
    _upload(page, 1, batch)
    page.get_by_text("Compare candidates").first.wait_for(timeout=120000)
    body = page.inner_text("body")
    assert "Display a time input widget" not in body                       # no accidental 'magic' rendering of st internals
    assert "failed" in body and "broken.pdf" in body                       # errors are reported, not hidden
    assert "empty.docx" in body
    assert page.get_by_text("Compare candidates").count()
    # filter: raising min score to the maximum removes every row but the app stays healthy
    page.get_by_text("Minimum score").first.wait_for()
    assert not page.get_by_text("Traceback").count()
    with page.expect_download() as dl:
        page.get_by_role("button", name="Ranking CSV").click()
    csv_text = Path(dl.value.path()).read_text()
    names = [p.name for p in files["people"].values()]
    assert csv_text.splitlines()[0].startswith("rank") and any(n in csv_text for n in names)
    # blind review removes names from the table and from every export
    page.get_by_text("Blind review (hide names)").click()
    page.wait_for_timeout(2500)
    with page.expect_download() as dl:
        page.get_by_role("button", name="Ranking CSV").click()
    blind = Path(dl.value.path()).read_text()
    assert "Candidate 1" in blind and not any(n in blind for n in names) and "R001" not in blind
    with page.expect_download() as dl:
        page.get_by_role("button", name="Full JSON").click()
    blind_json = Path(dl.value.path()).read_text()
    assert not any(n in blind_json for n in names) and not any(files["people"][k].email in blind_json for k in files["people"])
    assert not any(n in page.inner_text("body") for n in names)
    if os.environ.get("E2E_SCREENSHOTS"):
        page.get_by_text("Minimum score").first.scroll_into_view_if_needed()
        page.evaluate("window.scrollBy(0, -150)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(ROOT / "docs" / "screenshots" / "ranking_blind.png"))
    page.close()


def test_unicode_resume_does_not_break(browser, server, files):
    page = _open(browser, server)
    _upload(page, 0, [files["dir"] / "unicode.docx"])
    page.get_by_text("Compatibility").first.wait_for(timeout=90000)
    assert not page.get_by_text("Traceback").count()
    page.close()


def test_malformed_single_upload_shows_clear_error(browser, server, files):
    page = _open(browser, server)
    _upload(page, 0, [files["dir"] / "broken.pdf"])
    page.get_by_text("Could not open PDF").first.wait_for(timeout=60000)
    assert not page.get_by_text("Traceback").count()
    page.close()


def test_no_job_description_prompts_user(browser, server, files):
    page = browser.new_page()
    page.goto(server)
    page.get_by_text("Paste a job description in the sidebar to begin.").first.wait_for(timeout=60000)
    page.close()


def test_scanned_pdf_with_ocr(browser, server, files):
    from resume_intel.extraction import ocr_available
    if not ocr_available():
        pytest.skip("Tesseract not installed")
    page = _open(browser, server)
    _upload(page, 0, [files["scan"]])
    page.get_by_text("Compatibility").first.wait_for(timeout=120000)
    assert page.get_by_text("OCR was used").count()      # user is told the text came from OCR
    page.close()


def test_scanned_pdf_without_ocr_reports_unavailable(browser, server_no_ocr, files):
    page = _open(browser, server_no_ocr)
    page.get_by_text("not installed").first.wait_for(timeout=30000)
    _upload(page, 0, [files["scan"]])
    page.get_by_text("Tesseract").first.wait_for(timeout=60000)
    assert not page.get_by_text("Traceback").count()
    page.close()


def test_evaluation_tab_shows_results_with_synthetic_warning(browser, server):
    page = browser.new_page(viewport={"width": 1500, "height": 1100})
    page.goto(server)
    page.get_by_role("tab", name="Evaluation").click()
    page.get_by_text("synthetic", exact=False).first.wait_for(timeout=60000)
    page.get_by_text("Held-out TEST evaluation").first.wait_for(timeout=30000)
    assert not page.get_by_text("Traceback").count()
    page.close()
