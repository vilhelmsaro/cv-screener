"""Deterministic tests for splitting CV text into section and entry chunks."""
import pymupdf
import pytest

from cv_screener.index import pdf_text
from cv_screener.store import Chunk, chunk_cv

SIDEBAR_CV = """CONTACT

jane@example.com

SKILLS

Python, SQL

Jane Doe

Data Engineer

Builds pipelines.

Experience

Data Engineer
Feb 2022 – Present
Acme · Berlin

Built Spark jobs.

Moved ETL to Airflow.

Junior Analyst
2019 – 2021
Beta · Berlin

Wrote SQL reports.

Education

Frontend Bootcamp (React,
JavaScript)

Apr 2018 – Sep
2018
Code School · Berlin"""


def test_splits_by_section_and_entry():
    assert chunk_cv(SIDEBAR_CV, name="Jane Doe") == [
        Chunk("contact", "Contact\njane@example.com"),
        Chunk("skills", "Skills\nPython, SQL"),
        Chunk("header", "Jane Doe\nData Engineer\nBuilds pipelines."),  # not glued to Skills
        Chunk("experience", "Experience\nData Engineer\nFeb 2022 – Present\nAcme · Berlin\nBuilt Spark jobs.\n"
                            "Moved ETL to Airflow."),
        Chunk("experience", "Experience\nJunior Analyst\n2019 – 2021\nBeta · Berlin\nWrote SQL reports."),
        Chunk("education", "Education\nFrontend Bootcamp (React,\nJavaScript)\nApr 2018 – Sep\n2018\n"
                           "Code School · Berlin"),
    ]


@pytest.mark.parametrize("extracted_name", ["Jane Doe", "jane doe", "Jane", "Jane Q Doe"])
def test_name_match_tolerates_case_and_partial_names(extracted_name):
    text = "Skills\n\nPython\n\nJane Q. Doe\n\nSummary text."
    assert chunk_cv(text, name=extracted_name)[-1].section == "header"


def test_name_match_ignores_accents():
    text = "Skills\n\nPython\n\nLucía Fernández Ortega\n\nSummary text."
    assert chunk_cv(text, name="Lucia Fernandez")[-1] == Chunk("header", "Lucía Fernández Ortega\nSummary text.")


def test_name_inside_a_long_line_does_not_start_a_section():
    text = "Projects\n\nJane Doe Consulting website rebuilt for three local clients in Berlin"
    assert [c.section for c in chunk_cv(text, name="Jane Doe")] == ["projects"]


@pytest.mark.parametrize("heading, section", [
    ("SKILLS", "skills"), ("Technical Skills:", "skills"), ("Work  History", "experience"),
    ("PROFESSIONAL EXPERIENCE", "experience"), ("Hobbies", "interests"), ("Certificates", "certifications"),
])
def test_heading_variants(heading, section):
    assert chunk_cv(f"{heading}\n\nsome content")[0].section == section


@pytest.mark.parametrize("dates", ["Feb 2022 – Present", "2016 – 2019", "03/2019 - 06/2022", "01.2020 – 05.2022",
                                   "Sept. 2018 — Jan 2020", "2021-now"])
def test_date_formats_start_a_new_entry(dates):
    text = f"Experience\n\nFirst Job\n2010 – 2012\nA\n\nSecond Job\n{dates}\nB"
    assert len(chunk_cv(text)) == 2


def test_long_bullet_starting_with_years_does_not_start_an_entry():
    text = "Experience\n\nEngineer\n2018 – 2024\nAcme\n\n2019–2021 led the migration of all billing services to Kafka"
    assert len(chunk_cv(text)) == 1


def test_heading_followed_by_heading_makes_no_empty_chunk():
    assert chunk_cv("Projects\n\nInterests\n\nChess") == [Chunk("interests", "Interests\nChess")]


def test_text_before_any_heading_is_the_header():
    assert chunk_cv("Jane Doe\nData Engineer\n\nSkills\n\nPython")[0] == Chunk("header", "Jane Doe\nData Engineer")


def test_bullets_split_by_a_page_break_stay_in_their_job():
    text = "Experience\n\nEngineer\n2018 – 2024\nAcme\n\nBullet on page one.\n\nBullet continued on page two."
    assert chunk_cv(text) == [Chunk("experience", "Experience\nEngineer\n2018 – 2024\nAcme\n"
                                                  "Bullet on page one.\nBullet continued on page two.")]


def test_oversized_entry_is_split_and_keeps_its_title():
    text = "Experience\n\nLead\n2020 – 2024\nAcme\n\n" + "\n".join(f"Bullet {i} " + "x" * 50 for i in range(20))
    chunks = chunk_cv(text, max_chars=400)
    assert len(chunks) > 1
    assert all(c.section == "experience" and c.text.startswith("Experience\nLead") and len(c.text) <= 420
               for c in chunks)


def test_no_text_is_lost():
    chunks = chunk_cv(SIDEBAR_CV, name="Jane Doe")
    # Each non-header chunk starts with its printed heading; the rest is the CV text itself.
    chunk_lines = [line for c in chunks for line in c.text.splitlines()[c.section != "header":]]
    headings = {"CONTACT", "SKILLS", "Experience", "Education"}
    source_lines = [line for line in SIDEBAR_CV.splitlines() if line.strip() and line not in headings]
    assert sorted(chunk_lines) == sorted(source_lines)


def test_pdf_text_keeps_layout_blocks(tmp_path):
    path = tmp_path / "cv.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 60, 520, 90), "Experience")
    # The built-in PDF font has no en dash, so this also covers plain-hyphen date ranges.
    page.insert_textbox(pymupdf.Rect(72, 120, 520, 180), "Data Engineer\nFeb 2022 - Present\nAcme, Berlin")
    page.insert_textbox(pymupdf.Rect(72, 220, 520, 260), "Built Spark jobs.")
    page.insert_text((72, 760), "•")
    doc.save(path)
    text = pdf_text(path)
    assert "•" not in text
    assert chunk_cv(text) == [Chunk("experience", "Experience\nData Engineer\nFeb 2022 - Present\nAcme, Berlin\n"
                                                  "Built Spark jobs.")]
