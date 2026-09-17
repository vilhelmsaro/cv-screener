# NOTES

## Eval results

`cvs eval`, first complete run, 2026-09-17, unedited (also in `evals/last_run.txt`).
Models: agent `openai/gpt-4.1-mini`, embeddings `openai/text-embedding-3-small`.

```
[PASS] python_experience: Who has experience with Python?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Priya Raman
    ok  names Lucía Fernández Ortega
    ok  names Olumide Adeyemi
    ok  does not name Fatima Zahra El Amrani
[PASS] spanish_speakers: Which candidates speak Spanish?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Lucía Fernández Ortega
    ok  names Mateo Rojas Quintero
    ok  does not name Chen Wei
    ok  does not name Johannes Becker
    ok  does not name Priya Raman
[FAIL] senior_ml_fit: Who would be the best fit for a senior ML role?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    XX  names Priya Raman
    ok  does not name Fatima Zahra El Amrani
    ok  does not name Anna Petrosyan
    answer: No candidates matching the seniority level with Machine Learning skill and English language were found in the dataset. Thus, there is no best fit for a senior ML role available in the current candidate set.
[PASS] summarize_person: Summarize the profile of Johannes Becker
    ok  used a tool
    ok  called get_candidate
    ok  grounded (names came from tools)
    ok  names Johannes Becker
    ok  mentions key facts
[PASS] country_filter: Which candidates are based in Germany?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Johannes Becker
    ok  does not name Lucía Fernández Ortega
    ok  does not name Tomasz Nowak
[PASS] no_match_japanese: Which candidates speak Japanese?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names nobody
    ok  says no match
[PASS] no_match_pilot: Who has worked as a commercial airline pilot?
    ok  used a tool
    ok  grounded (names came from tools)
    ok  names nobody
    ok  says no match
[PASS] unknown_person: Summarize the profile of Maria Gonzalez
    ok  used a tool
    ok  grounded (names came from tools)
    ok  names nobody
    ok  says no match
[PASS] semantic_recommender: Who has built recommender systems in production?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Priya Raman
[PASS] filter_plus_semantic: Which Spanish speakers have Kubernetes experience?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Mateo Rojas Quintero
    ok  does not name Lucía Fernández Ortega
    ok  does not name Chen Wei
[PASS] partial_name: Summarize the profile of lucia fernandez
    ok  used a tool
    ok  called get_candidate
    ok  grounded (names came from tools)
    ok  names Lucía Fernández Ortega

TOTAL: 10/11 cases passed
```

### Second run, after the two fixes below (2026-09-17), unedited

```
[PASS] python_experience: Who has experience with Python?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Priya Raman
    ok  names Lucía Fernández Ortega
    ok  names Olumide Adeyemi
    ok  does not name Fatima Zahra El Amrani
[PASS] spanish_speakers: Which candidates speak Spanish?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Lucía Fernández Ortega
    ok  names Mateo Rojas Quintero
    ok  does not name Chen Wei
    ok  does not name Johannes Becker
    ok  does not name Priya Raman
[PASS] senior_ml_fit: Who would be the best fit for a senior ML role?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Priya Raman
    ok  does not name Fatima Zahra El Amrani
    ok  does not name Anna Petrosyan
[PASS] summarize_person: Summarize the profile of Johannes Becker
    ok  used a tool
    ok  called get_candidate
    ok  grounded (names came from tools)
    ok  names Johannes Becker
    ok  mentions key facts
[PASS] country_filter: Which candidates are based in Germany?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Johannes Becker
    ok  does not name Lucía Fernández Ortega
    ok  does not name Tomasz Nowak
[PASS] no_match_japanese: Which candidates speak Japanese?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names nobody
    ok  says no match
[PASS] no_match_pilot: Who has worked as a commercial airline pilot?
    ok  used a tool
    ok  grounded (names came from tools)
    ok  names nobody
    ok  says no match
[PASS] unknown_person: Summarize the profile of Maria Gonzalez
    ok  used a tool
    ok  grounded (names came from tools)
    ok  names nobody
    ok  says no match
[PASS] semantic_recommender: Who has built recommender systems in production?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Priya Raman
[PASS] filter_plus_semantic: Which Spanish speakers have Kubernetes experience?
    ok  used a tool
    ok  called search_candidates
    ok  grounded (names came from tools)
    ok  names Mateo Rojas Quintero
    ok  does not name Lucía Fernández Ortega
    ok  does not name Chen Wei
[PASS] partial_name: Summarize the profile of lucia fernandez
    ok  used a tool
    ok  called get_candidate
    ok  grounded (names came from tools)
    ok  names Lucía Fernández Ortega

TOTAL: 11/11 cases passed
```

### Fresh-clone check (2026-09-17)

The repo was cloned to a new directory, built, and run through the README from scratch: `generate` wrote
12 new CVs with photos, `index` reported coverage OK, the offline tests passed on that new data, and
`cvs eval` again scored **11/11**, so the cases are not tuned to one generated dataset. That run also
caught a real bug: the generator emitted "Git, Jenkins" as one skill item, which the page renders
identically to two skills, fixed by splitting skill items on commas outside brackets.

## Not done / known issues
- The first run failed `senior_ml_fit`: the agent filtered on the exact skill "Machine Learning", which no
  CV prints, plus a language filter the question never asked for, got an empty list and answered that
  nobody matches. Two fixes followed, and the second run passes 11/11:
  1. `search_candidates` now explains an empty result ("filters are exact, retry with the words in
     `query`") instead of returning a bare empty list, and the agent is told to filter only on what the
     question states and never by seniority for "best fit" questions.
  2. A semantic hit must score within 70% of the best hit and above a floor. Without it, "Which Spanish
     speakers have Kubernetes experience?" also returned two candidates with no Kubernetes at all.
- The relevance thresholds are calibrated for `openai/text-embedding-3-small` on these 12 CVs (real
  matches 0.44-0.57, unrelated 0.34 or less). Another embedding model would need them re-measured.
- `get_candidate` returns the first candidate whose name words match; with two similar names it would
  silently pick one. No two candidates share name words in this dataset.
- Photo generation depends on the chosen OpenRouter image model. A failed candidate is skipped and listed at the end;
  rerun with `cvs generate --only <id>`.
- Skill matching via filters is exact after normalization ("PyTorch" vs "Pytorch" ok, "Java 17" and
  "Docker (basic)" also match "Java" / "Docker", "ML" vs "machine learning" not); the agent is instructed to
  fall back to semantic search.
- Entries are separated by a date range on its own line. A CV that prints "Analyst, Acme (2020 - 2022)" on
  one line still indexes and keeps its fields, but its jobs land in one chunk instead of one each.
- Field rules assume English headings and common CV conventions (a "Skills" section, "N years of experience"
  or dated jobs, "City, Country" next to the email). All 12 generated CVs parse fully without the LLM; other
  CVs may leave a field empty rather than get a wrong value.
- Generation checks each CV against its seed with the same rules indexing uses (level from the current title,
  years stated in the summary, spoken languages) and sends mismatches back to the model to fix. After the
  retries run out, that candidate fails and is listed at the end of `cvs generate`.
- No keyword (BM25) ranking; search is metadata filters + semantic similarity over chunks. The profile-level
  embedding is stored but not queried yet.
- Chunking relies on common English section headings and date ranges ("Feb 2022 – Present"). A CV with
  unusual headings falls back to fewer, larger chunks rather than failing.

## Next steps
- BM25 + vector fusion, reranker for "best fit" questions.
- LLM-as-judge eval for answer quality on top of the deterministic checks.
- Skill synonym map / taxonomy for filters.
- Web UI or public deploy.
