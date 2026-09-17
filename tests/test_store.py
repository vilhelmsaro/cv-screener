import pymupdf

from cv_screener.index import pdf_text
from cv_screener.store import CandidateStore, chunk_text, norm

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


def test_pdf_blocks_become_separate_chunks(tmp_path):
    path = tmp_path / "cv.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    for i, y in enumerate(range(80, 700, 60)):
        page.insert_textbox(pymupdf.Rect(72, y, 520, y + 50), f"Job {i}: " + "shipped features " * 12)
    page.insert_text((72, 760), "•")
    doc.save(path)
    text = pdf_text(path)
    assert "•" not in text
    chunks = chunk_text(text)
    assert len(chunks) > 1 and all(len(c) < 1200 for c in chunks)


def test_helpers():
    assert norm("C++") == "cplusplus" and norm("Node.js") == "nodejs" and norm("Español") == "espanol"
    assert CandidateStore.build_where() is None
    assert len(chunk_text("a\n\n" * 5, size=1)) == 5
