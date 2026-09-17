from cv_screener.store import CandidateStore, chunk_text, norm


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


def test_get_candidate_is_accent_and_partial_insensitive(store):
    assert store.get("lucia fernandez")["fields"]["name"] == "Lucía Fernández Ortega"
    assert store.get("Maria Gonzalez") is None


def test_helpers():
    assert norm("C++") == "cplusplus" and norm("Node.js") == "nodejs" and norm("Español") == "espanol"
    assert CandidateStore.build_where() is None
    assert len(chunk_text("a\n\n" * 5, size=1)) == 5
