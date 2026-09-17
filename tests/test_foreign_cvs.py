"""Indexing on CVs our own templates never produced.

Our three templates could quietly become the only layout the rules understand, so these fixtures use
other conventions: different headings, dates on the title line, dash bullets, a two-column header.
Real CVs vary more than this; the point is that a different layout still indexes, and never loses text.
"""
from datetime import date

import pymupdf
import pytest

from cv_screener.index import pdf_name, read_cv
from cv_screener.store import chunk_cv, missing_sections

TODAY = date(2026, 9, 17)

DATE_ON_OWN_LINE = [  # a common single-column layout, headings we never render
    (22, "Marta Silva"),
    (10, "Lisbon, Portugal | marta.silva@example.pt | +351 912 345 678"),
    (12, "PROFILE"),
    (10, "Backend engineer with 9 years of experience in payments."),
    (12, "EMPLOYMENT HISTORY"),
    (10, "Backend Engineer\nMar 2019 - Present\nStripe, Lisbon"),
    (10, "- Built the refunds service.\n- Cut latency by 40%."),
    (10, "Software Developer\nJan 2016 - Feb 2019\nFeedzai, Lisbon"),
    (10, "- Maintained the fraud rules engine."),
    (12, "EDUCATION"),
    (10, "BSc Informatics\n2012 - 2015\nUniversity of Lisbon"),
    (12, "KEY SKILLS"),
    (10, "Python, Django, PostgreSQL, Kafka"),
    (12, "LANGUAGES"),
    (10, "Portuguese (Native), English (C1)"),
]

DATES_ON_THE_TITLE_LINE = [  # the harder case: nothing marks where an entry starts
    (22, "Ivan Horvat"),
    (10, "Zagreb, Croatia | ivan.horvat@example.hr"),
    (12, "Summary"),
    (10, "Data analyst with 4 years of experience in retail analytics."),
    (12, "Work Experience"),
    (10, "Data Analyst, Infobip (Feb 2022 - Present)"),
    (10, "- Built dashboards for the sales team."),
    (10, "Junior Analyst, Rimac (Sep 2020 - Jan 2022)"),
    (10, "- Prepared weekly reports."),
    (12, "Education"),
    (10, "BSc Mathematics, University of Zagreb (2016 - 2020)"),
    (12, "Skills"),
    (10, "SQL, Python, Tableau"),
    (12, "Languages"),
    (10, "Croatian (Native), English (B2)"),
]


def write_cv(path, blocks) -> str:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    for size, text in blocks:
        height = size * 1.7 * (text.count("\n") + 1) + 8  # a box too small for the font drops the text
        assert page.insert_textbox(pymupdf.Rect(60, y, 540, y + height), text, fontsize=size) >= 0
        y += height + 10
    doc.save(path)
    return path


def all_words(text: str) -> set[str]:
    return {w.casefold() for w in text.replace("\n", " ").split() if any(c.isalnum() for c in w)}


@pytest.fixture(params=[DATE_ON_OWN_LINE, DATES_ON_THE_TITLE_LINE], ids=["dates_own_line", "dates_inline"])
def foreign_cv(request, tmp_path):
    return write_cv(tmp_path / "cv.pdf", request.param)


def test_foreign_cv_indexes_without_losing_text(foreign_cv):
    text, fields, unresolved = read_cv(foreign_cv, today=TODAY)
    chunks = chunk_cv(text, name=fields.full_name)
    assert missing_sections({c.section: 1 for c in chunks}) == []
    assert unresolved == []
    # Nothing from the PDF may disappear: chunk text covers every word except the printed headings.
    assert all_words(text) - {w for c in chunks for w in all_words(c.text)} == set()


def test_foreign_cv_with_dates_on_their_own_line(tmp_path):
    _, fields, _ = read_cv(write_cv(tmp_path / "a.pdf", DATE_ON_OWN_LINE), today=TODAY)
    assert (fields.full_name, fields.city, fields.country) == ("Marta Silva", "Lisbon", "Portugal")
    assert fields.current_title == "Backend Engineer"
    assert (fields.seniority, fields.years_experience) == ("senior", 9)
    assert fields.skills == ["Python", "Django", "PostgreSQL", "Kafka"]
    assert fields.languages == ["Portuguese", "English"]
    assert fields.highest_education == "BSc Informatics"
    text, _, _ = read_cv(write_cv(tmp_path / "a.pdf", DATE_ON_OWN_LINE), today=TODAY)
    jobs = [c for c in chunk_cv(text, name="Marta Silva") if c.section == "experience"]
    assert len(jobs) == 2 and "refunds service" in jobs[0].text and "fraud rules engine" in jobs[1].text


def test_foreign_cv_with_dates_on_the_title_line_degrades_gracefully(tmp_path):
    path = write_cv(tmp_path / "b.pdf", DATES_ON_THE_TITLE_LINE)
    text, fields, _ = read_cv(path, today=TODAY)
    assert (fields.full_name, fields.city, fields.country) == ("Ivan Horvat", "Zagreb", "Croatia")
    assert fields.current_title == "Data Analyst, Infobip"  # dates stripped from the title line
    assert (fields.seniority, fields.years_experience) == ("mid", 4)
    assert fields.skills == ["SQL", "Python", "Tableau"]
    assert fields.languages == ["Croatian", "English"]
    # Known limit: without a date line the jobs are not separated, so the section stays one chunk.
    jobs = [c for c in chunk_cv(text, name="Ivan Horvat") if c.section == "experience"]
    assert len(jobs) == 1 and "Infobip" in jobs[0].text and "Rimac" in jobs[0].text


def test_pdf_name_reads_the_foreign_header(foreign_cv):
    assert pdf_name(foreign_cv) in {"Marta Silva", "Ivan Horvat"}
