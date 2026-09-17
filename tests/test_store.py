import uuid
from collections import Counter

import chromadb

from cv_screener.index import check_coverage
from cv_screener.store import CandidateStore, missing_sections, norm

from .fakes import FakeEmbedder, make_fields


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


def test_helpers():
    assert norm("C++") == "cplusplus" and norm("Node.js") == "nodejs" and norm("Español") == "espanol"
    assert CandidateStore.build_where() is None


FULL_CV = """Johannes Becker
Senior Embedded Engineer

Professional Experience

Senior Embedded Engineer
Apr 2020 – Present
Continental · Regensburg

Firmware for ECUs.

Education

Dipl.-Ing. Electrical Engineering
2004 – 2010
TU München

Skills

C, C++, AUTOSAR

Languages

German — Native"""


def test_stored_sections_match_what_was_sent(store):
    sent = store.upsert("c08", make_fields("Johannes Becker", "Senior Embedded Engineer", "Germany", "senior", 14,
                                           ["C++"], ["German"]), FULL_CV)
    assert sent == {"header": 1, "experience": 1, "education": 1, "skills": 1, "languages": 1}
    stored = store.section_counts()["c08"]
    assert stored["name"] == "Johannes Becker" and stored["sections"] == sent
    assert missing_sections(stored["sections"]) == []
    sections = store.chunks.get(where={"candidate_id": "c08"}, include=["metadatas"])["metadatas"]
    assert sorted(m["section"] for m in sections) == sorted(sent)


def test_coverage_flags_missing_sections_and_lost_chunks(store):
    # The fixture CVs are one-paragraph texts with no headings, so they only have a header section.
    problems = check_coverage(store, sent=None)
    assert "c01: no 'experience' chunks" in problems
    assert not any(p.startswith("c01: sent") for p in problems)

    sent = {"c01": store.section_counts()["c01"]["sections"] + Counter(experience=2)}
    assert any(p.startswith("c01: sent") for p in check_coverage(store, sent=sent))


def test_coverage_fails_on_empty_store():
    empty = CandidateStore(FakeEmbedder(), client=chromadb.EphemeralClient(), prefix=f"e{uuid.uuid4().hex[:8]}")
    assert check_coverage(empty) == ["store is empty"]


def test_coverage_passes_for_complete_cv(store):
    for cid in ("c01", "c04"):
        store.chunks.delete(where={"candidate_id": cid})
    sent = {"c08": store.upsert("c08", make_fields("Johannes Becker", "Senior Embedded Engineer", "Germany",
                                                   "senior", 14, ["C++"], ["German"]), FULL_CV)}
    assert check_coverage(store, sent=sent) == []


def test_field_search_returns_evidence_fields(store):
    hit = store.search(languages=["Spanish"])[0]
    assert "Spanish" in hit.languages and "Python" in hit.skills
