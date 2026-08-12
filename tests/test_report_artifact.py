"""Structural checks for the editable Part B Word report artifact."""

from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "report.docx"


def _document_xml() -> str:
    with ZipFile(REPORT) as archive:
        return archive.read("word/document.xml").decode("utf-8")


def test_report_docx_is_valid_word_package():
    assert REPORT.exists()
    with ZipFile(REPORT) as archive:
        names = set(archive.namelist())
    assert "word/document.xml" in names
    assert "word/styles.xml" in names


def test_report_contains_required_sections_and_exhibits():
    xml = _document_xml()
    for text in (
            "Executive Summary",
            "Fund design and walk-forward testing",
            "Coverage adjustment helps control a weak signal",
            "Reflection and recommendations",
            "Appendix A",
            "Exhibit A1",
            "Exhibit A7",
    ):
        assert text in xml


def test_report_embeds_all_six_required_figure_files():
    with ZipFile(REPORT) as archive:
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
    assert len(media) == 6

