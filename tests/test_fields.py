"""Deterministic tests for reading structured fields from CV text with rules."""
from datetime import date

import pymupdf
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from cv_screener import index
from cv_screener.countries import canonical_country
from cv_screener.fields import (
    extract_fields,
    find_location,
    highest_degree,
    parse_languages,
    parse_skills,
    seniority_for,
    split_location,
    years_from_dates,
    years_from_text,
)
from cv_screener.store import Chunk, chunk_cv, skill_key_variants

TODAY = date(2026, 9, 17)


def test_skills_groups_wrapped_lines_and_parentheses():
    lines = ["Frontend: React, TypeScript,", "HTML/CSS, Tailwind CSS", "Languages & Frameworks:", "Python, Cross-",
             "functional work, AWS (EC2, S3), Docker (basic)", "Other: Agile/", "Scrum"]
    assert parse_skills(lines) == ["React", "TypeScript", "HTML/CSS", "Tailwind CSS", "Python",
                                   "Cross-functional work", "AWS (EC2, S3)", "Docker (basic)", "Agile/Scrum"]


def test_skills_in_a_labeled_group_continue_across_capitalised_wraps():
    assert parse_skills(["Testing: Selenium", "WebDriver, pytest"]) == ["Selenium WebDriver", "pytest"]


def test_skills_without_group_labels_and_duplicates():
    assert parse_skills(["Python, SQL; python", "Docker"]) == ["Python", "SQL", "Docker"]


@pytest.mark.parametrize("lines, expected", [
    (["Spanish — Native", "English — C1", "Spanish — A2 - Basic"], ["Spanish", "English"]),
    (["Polish (Native), English (B2), German (A2)"], ["Polish", "English", "German"]),
    (["Portuguese —", "B1"], ["Portuguese"]),  # wrapped level is not a language
    (["French: fluent"], ["French"]),
])
def test_languages(lines, expected):
    assert parse_languages(lines) == expected


def test_location_from_contact_line_not_headline():
    header = Chunk("header", "Jane Doe\nData Scientist | Experimentation, Forecasting\n"
                             "Madrid, Spain · jane@example.com · +34 611 234 567")
    assert find_location([header]) == "Madrid, Spain"


def test_location_from_sidebar_contact_skips_wrapped_email_and_links():
    contact = Chunk("contact", "Contact\nanna.dev@gmail.co\nm\n+374 77 45 62 19\nYerevan, Armenia\nlinkedin.com/in/x")
    assert find_location([contact]) == "Yerevan, Armenia"


def test_location_without_comma_must_be_a_country():
    header = Chunk("header", "Chen Wei\niOS Tech Lead\nSingapore | chen@example.com | +65 8734 2916")
    assert find_location([header]) == "Singapore"
    assert find_location([Chunk("header", "Jane\nRemote | jane@example.com")]) is None


@pytest.mark.parametrize("location, expected", [
    ("Bogotá, Colombia (Remote)", ("Bogotá", "Colombia")),
    ("Austin, Texas, USA", ("Austin", "United States")),
    ("London, UK", ("London", "United Kingdom")),
    ("Singapore", ("Singapore", "Singapore")),
    ("Springfield, Nowhere", ("Springfield", None)),
    (None, ("", None)),
])
def test_split_location(location, expected):
    assert split_location(location) == expected


def test_canonical_country():
    assert canonical_country("germany") == "Germany" and canonical_country("USA") == "United States"
    assert canonical_country("Texas") is None and canonical_country("") is None


@pytest.mark.parametrize("text, years", [
    ("Senior Data Scientist with 8 years of experience", 8), ("1 year of hands-on experience", 1),
    ("over ten years in fintech", 10), ("12+ yrs building apps", 12), ("Loves building apps", None),
])
def test_years_from_text(text, years):
    assert years_from_text(text) == years


def test_years_from_dates_merges_overlaps_and_reads_present():
    jobs = [Chunk("experience", "Experience\nLead\nJan 2022 – Present\nAcme"),         # 4y 8m to Sep 2026
            Chunk("experience", "Experience\nConsultant\nJun 2021 – Jun 2022\nSelf"),  # overlaps by 5 months
            Chunk("experience", "Experience\nIntern\n01.2018 – 07.2018\nBeta"),        # 6 months
            Chunk("experience", "Experience\nBootcamp\nApr 2016 – Sep\n2016\nSchool")]  # wrapped range, 5 months
    assert years_from_dates(jobs, TODAY) == 6  # 56 + 7 + 6 + 5 = 74 months


@pytest.mark.parametrize("title, years, level", [
    ("Senior Engineering Manager", 15, "manager"), ("iOS Tech Lead", 10, "lead"),
    ("Staff Machine Learning Engineer", 11, "staff"), ("Junior ML Researcher", 1, "junior"),
    ("Design Intern", 0, "intern"), ("Data Engineer", 4, "mid"), ("Data Engineer", 1, "junior"),
    ("Backend Developer", 9, "senior"), ("Sr. Developer", 3, "senior"),
])
def test_seniority(title, years, level):
    assert seniority_for(title, years) == level


def test_highest_degree_ranks_by_level_not_order():
    education = [Chunk("education", "Education\nFrontend Bootcamp\nApr 2024 – Sep 2024\nCode School"),
                 Chunk("education", "Education\nBA in Linguistics\n2015 – 2019\nYerevan State University"),
                 Chunk("education", "Education\nMSc Statistics\n2019 – 2021\nUCM")]
    assert highest_degree(education) == "MSc Statistics"
    assert highest_degree(education[:1]) == "Frontend Bootcamp"
    assert highest_degree([]) == ""


def test_skill_key_variants_match_plain_names():
    assert {"java", "java_17"} <= skill_key_variants("Java 17")
    assert "docker" in skill_key_variants("Docker (basic)")
    assert {"html", "css"} <= skill_key_variants("HTML/CSS")
    assert skill_key_variants("A/B testing") == {"a_b_testing"}


CV = """CONTACT

jane@example.com

Berlin, Germany

SKILLS

Languages: Python, SQL

LANGUAGES

German — Native

English — C1

Jane Doe

Senior Data Engineer | Pipelines

Senior Data Engineer with 7 years of experience in data platforms.

Experience

Senior Data Engineer
Feb 2022 – Present
Acme · Berlin

Built Spark jobs.

Education

BSc Computer Science
2012 – 2016
TU Berlin"""


def test_extract_fields_end_to_end():
    fields, unresolved = extract_fields(chunk_cv(CV, name="Jane Doe"), "Jane Doe", TODAY)
    assert unresolved == []
    assert fields.model_dump() == {
        "full_name": "Jane Doe", "current_title": "Senior Data Engineer", "city": "Berlin", "country": "Germany",
        "seniority": "senior", "years_experience": 7, "skills": ["Python", "SQL"],
        "languages": ["German", "English"], "highest_education": "BSc Computer Science",
        "summary": "Senior Data Engineer | Pipelines Senior Data Engineer with 7 years of experience in data "
                   "platforms.",
    }


def test_extract_fields_reports_unreadable_country():
    text = CV.replace("Berlin, Germany", "Remote")
    fields, unresolved = extract_fields(chunk_cv(text, name="Jane Doe"), "Jane Doe", TODAY)
    assert unresolved == ["country"] and fields.country == ""


def test_llm_country_fallback_keeps_only_real_countries(monkeypatch):
    def fake_agent(answer):
        return lambda *args, **kwargs: Agent(TestModel(custom_output_text=answer), output_type=str)

    monkeypatch.setattr(index, "structured_agent", fake_agent("germany"))
    assert index._country_from_llm(CV, "Jane Doe") == "Germany"
    monkeypatch.setattr(index, "structured_agent", fake_agent("Atlantis"))
    assert index._country_from_llm(CV, "Jane Doe") == ""


def test_pdf_name_is_the_largest_text(tmp_path):
    path = tmp_path / "cv.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "CONTACT", fontsize=9)
    page.insert_text((72, 120), "Jane Doe", fontsize=22)
    page.insert_text((72, 150), "Data Engineer", fontsize=11)
    doc.save(path)
    assert index.pdf_name(path) == "Jane Doe"
