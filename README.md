# CV Screener

Generates realistic synthetic CVs (PDF with AI-generated photos), indexes them into a vector store with
structured fields, and answers questions through a CLI agent that searches the index with tools.

**Stack:** Python 3.12 · PydanticAI (agent + structured outputs) · OpenRouter (LLM, images, embeddings)
· ChromaDB · WeasyPrint · PyMuPDF · Typer.

## Quick start (Docker)

Requirements: Docker with Compose, an [OpenRouter](https://openrouter.ai/keys) key. A full run (12 CVs
with photos, indexing, one eval run) cost about $1.20 with the models in `.env.example`; re-runs are
cached and cost nothing.

```bash
git clone <repo-url> cv-screener && cd cv-screener
cp .env.example .env            # put your OPENROUTER_API_KEY in .env
docker compose build

docker compose run --rm app generate   # 1. 12 candidates -> data/cvs/*.pdf
docker compose run --rm app index      # 2. PDFs -> fields + embeddings -> Chroma, prints coverage
docker compose run --rm app coverage   #    re-check what is stored, per CV section (no API calls)
docker compose run --rm app chat       # 3. interactive agent
docker compose run --rm app eval       # evals, prints pass/fail + total
docker compose run --rm --entrypoint pytest app   # tests, no API key needed
```

One-off question: `docker compose run --rm app ask "Which candidates speak Spanish?"`

## Local run (without Docker)

Requires Python 3.11+ and WeasyPrint system libraries
([install notes](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)).
With `CHROMA_HOST` empty, Chroma runs embedded in `data/chroma`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your key
cvs generate && cvs index && cvs chat
pytest
```

## How it works

1. **Generation** (`cv_screener/generate.py`). `seeds.py` fixes a diversity matrix: 12 people with different
   roles, levels, countries, stacks, education and languages. An LLM expands each seed into a full
   `CandidateProfile` (validated with Pydantic), an image model generates the headshot, and one of three
   HTML templates (`classic`, `modern` with a sidebar, `compact` US Letter) is rendered to PDF. Profiles and photos are cached, so
   re-runs are free (`--force` regenerates).
2. **Indexing** (`cv_screener/index.py`, `fields.py`, `store.py`). Text is extracted from the **PDFs** (not
   from the source JSON). Structured fields are read from the page with plain rules, so the same PDF always
   gives the same fields: the name is the largest text on page 1, skills and languages come from their
   sections, the current title from the first job, years of experience from the summary ("8 years of
   experience") or else the job dates, the country from the contact line (checked against the ISO country
   list), and seniority from title words or years. Only a country the rules cannot read goes to the LLM, and
   its answer is kept only if it is a real country. Two Chroma collections are written:
   `profiles` (one record per candidate) and `chunks` (one per CV section, and one per job or degree,
   found from section headings and date ranges in the PDF text; each is embedded with the candidate's
   name and title). Every record carries the fields as metadata, including normalized
   `skill_keys` / `lang_keys` lists for exact filtering (`{"skill_keys": {"$contains": "python"}}`), and
   each chunk has a `section` label (`header`, `experience`, `education`, `skills`, `languages`, ...).
   After writing, indexing reads the labels back from Chroma and fails if a CV is missing a required
   section or if fewer chunks were stored than sent; `cvs coverage` runs the same check on demand.
3. **Search.** `CandidateStore.search()` combines metadata filters (skills, languages, seniority, country,
   min years) with semantic similarity over chunks; with no query it is a pure field search.
4. **Agent** (`cv_screener/agent.py`). A PydanticAI agent with three tools: `search_candidates`,
   `get_candidate`, `list_filter_values`. It never receives the whole dataset; instructions require it to
   name only candidates returned by tools and to say when nobody matches.

## Evals and tests

- `evals/cases.yaml` has 11 cases: the four example questions, a field filter, semantic-only and
  filter-plus-semantic search, a partial name, and three no-match cases.
  Each checks tool usage, expected/forbidden names, and **grounding**: every candidate named in the answer
  must appear in tool results. Results are written to `evals/last_run.txt`.
- `tests/` runs offline with a hashing fake embedder, an in-memory Chroma, and PydanticAI's `TestModel`.
  `tests/test_chunking.py` pins the chunking rules and edge cases. `tests/test_foreign_cvs.py` indexes CVs
  in layouts our templates never produce. `tests/test_boundaries.py` keeps the read path from importing the
  seeds or the generated JSON, which a real deployment would not have. `tests/test_generated_cvs.py` checks the
  real PDFs in `data/cvs` (skipped until `generate` has run): every job and degree is exactly one chunk with
  all its bullets, and every extracted field matches the answer key (the generator's JSON and seeds, used
  only for checking). `tests/test_fields.py` covers each extraction rule on small inputs.

## Layout

```
cv_screener/  config, models, seeds, llm, generate, store, fields, parsing, countries, index, agent, cli, templates/
evals/        cases.yaml, run_evals.py
tests/        offline unit tests
data/         generated output (git-ignored)
```
