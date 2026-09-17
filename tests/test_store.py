import pymupdf

from cv_screener.index import pdf_text
from cv_screener.store import CandidateStore, chunk_cv, norm

from .fakes import make_fields


def test_field_filters_are_exact(store):
    names = {h.name for h in store.search(languages=["spanish"])}
    assert names == {"Lucía Fernández Ortega"}
    assert {h.name for h in store.search(skills=["c++"])} == {"Johannes Becker"}
    assert {h.name for h in store.search(min_years=5, seniority=["senior"])} == {"Lucía Fernández Ortega",
                                                                                 "Johannes Becker"}
    assert store.search(languages=["Japanese"]) == []


def test_semantic_search_ranks_relevant_candidate_first(store):
    hits = store.search(query="automotive firmware CAN bus")
    assert hits[0].name == "Johannes Becker"
    assert hits[0].snippets


def test_semantic_plus_filter(store):
    hits = store.search(query="dashboards", country="Armenia")
    assert [h.name for h in hits] == ["Anna Petrosyan"]
    assert hits[0].location == "X, Armenia"  # display value, not the normalized filter key


def test_reindex_drops_removed_skills(store):
    store.upsert("c08", make_fields("Johannes Becker", "Senior Embedded Engineer", "Germany", "senior", 14,
                                ["C"], ["German"]), "Johannes Becker\n\nFirmware in C.")
    assert store.search(skills=["C++"]) == []
    assert "Johannes Becker" not in {h.name for h in store.search(languages=["English"])}
    assert store.chunks.get(where={"candidate_id": "c08"})["documents"] == ["Johannes Becker\nFirmware in C."]


def test_get_candidate_is_accent_and_partial_insensitive(store):
    assert store.get("lucia fernandez")["fields"]["name"] == "Lucía Fernández Ortega"
    assert store.get("Becker")["fields"]["name"] == "Johannes Becker"
    assert store.get("Maria Gonzalez") is None
    assert store.get("an") is None and store.get("") is None


CV_TEXT = """CONTACT

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


def test_chunk_cv_splits_by_section_and_entry():
    chunks = chunk_cv(CV_TEXT, name="Jane Doe")
    assert chunks == [
        "Contact\njane@example.com",
        "Skills\nPython, SQL",
        "Jane Doe\nData Engineer\nBuilds pipelines.",  # sidebar layout: summary is not glued to Skills
        "Experience\nData Engineer\nFeb 2022 – Present\nAcme · Berlin\nBuilt Spark jobs.\nMoved ETL to Airflow.",
        "Experience\nJunior Analyst\n2019 – 2021\nBeta · Berlin\nWrote SQL reports.",
        "Education\nFrontend Bootcamp (React,\nJavaScript)\nApr 2018 – Sep\n2018\nCode School · Berlin",
    ]


def test_chunk_cv_splits_oversized_entry_and_keeps_title():
    text = "Experience\n\nLead\n2020 – 2024\nAcme\n\n" + "\n".join(f"Bullet {i} " + "x" * 50 for i in range(20))
    chunks = chunk_cv(text, max_chars=400)
    assert len(chunks) > 1
    assert all(c.startswith("Experience\nLead") and len(c) <= 420 for c in chunks)


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
    assert chunk_cv(text) == ["Experience\nData Engineer\nFeb 2022 - Present\nAcme, Berlin\nBuilt Spark jobs."]


def test_helpers():
    assert norm("C++") == "cplusplus" and norm("Node.js") == "nodejs" and norm("Español") == "espanol"
    assert CandidateStore.build_where() is None
