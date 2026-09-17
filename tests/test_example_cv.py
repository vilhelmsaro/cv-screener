"""The real CV in examples/ is the regression fixture for layouts a generated CV never produces.

Its PDF blocks end paragraphs with the next section's heading, its jobs put the dates on the title line,
and its spoken languages are a "Languages: ..." line instead of a section. Offline and deterministic.
"""
import uuid
from collections import Counter
from datetime import date
from pathlib import Path

import chromadb
import pytest

from cv_screener.index import read_cv
from cv_screener.store import CandidateStore, chunk_cv, missing_sections

from .fakes import FakeEmbedder

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.pdf"))
pytestmark = pytest.mark.skipif(not EXAMPLES, reason="no example CV")
CV = EXAMPLES[0]


@pytest.fixture(scope="module")
def parsed():
    text, fields, unresolved = read_cv(CV, today=date(2026, 9, 18))
    return text, fields, unresolved


def test_every_section_is_found(parsed):
    text, fields, _ = parsed
    counts = Counter(c.section for c in chunk_cv(text, name=fields.full_name))
    assert missing_sections(counts) == []
    assert counts["experience"] == 3  # three jobs, each its own chunk
    assert counts["projects"] >= 1 and counts["education"] == 1


def test_fields_are_read_from_the_page(parsed):
    _, fields, unresolved = parsed
    assert unresolved == []  # no LLM fallback needed
    assert fields.full_name == "SARO VILHELM YEKANIAN"
    assert fields.current_title == "Software Engineer"
    assert (fields.city, fields.country) == ("Yerevan", "Armenia")
    assert (fields.seniority, fields.years_experience) == ("mid", 5)
    assert fields.languages == ["Armenian", "English", "Russian"]
    assert {"Node.js", "TypeScript", "PostgreSQL", "NestJS"} <= set(fields.skills)
    assert fields.highest_education == "Bachelor of Science in Computer Science"


def test_it_indexes_and_is_searchable(parsed):
    text, fields, _ = parsed
    store = CandidateStore(FakeEmbedder(), client=chromadb.EphemeralClient(),
                           prefix=f"x{uuid.uuid4().hex[:8]}", score_floor=0)
    store.upsert(CV.stem, fields, text)
    assert [h.name for h in store.search(languages=["Armenian"])] == ["SARO VILHELM YEKANIAN"]
    assert [h.name for h in store.search(skills=["NestJS"])] == ["SARO VILHELM YEKANIAN"]
    assert store.get("saro yekanian")["fields"]["country"] == "Armenia"
