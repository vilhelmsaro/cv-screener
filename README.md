# CV Screener

Generates realistic synthetic CVs (PDF with AI-generated photos), indexes them into a vector store with
structured fields, and answers questions through a CLI agent that searches the index with tools.

**Stack:** Python 3.12 · PydanticAI (agent + structured outputs) · OpenRouter (LLM, images, embeddings)
· ChromaDB · WeasyPrint · PyMuPDF · Typer.

## Quick start (Docker)

Requirements: Docker with Compose, an [OpenRouter](https://openrouter.ai/keys) key (a full run costs well under $1).

```bash
git clone <repo-url> cv-screener && cd cv-screener
cp .env.example .env            # put your OPENROUTER_API_KEY in .env
docker compose build

docker compose run --rm app generate   # 1. 12 candidates -> data/cvs/*.pdf
docker compose run --rm app index      # 2. PDFs -> fields + embeddings -> Chroma
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
   HTML templates (classic, sidebar, compact US) is rendered to PDF. Profiles and photos are cached, so
   re-runs are free (`--force` regenerates).
2. **Indexing** (`cv_screener/index.py`, `store.py`). Text is extracted from the **PDFs** (not from the
   source JSON), an LLM extracts `ExtractedFields`, and two Chroma collections are written:
   `profiles` (one record per candidate) and `chunks` (groups of PDF text blocks, each embedded with the
   candidate's name and title). Every record carries the fields as metadata, including normalized
   `skill_keys` / `lang_keys` lists for exact filtering (`{"skill_keys": {"$contains": "python"}}`).
3. **Search.** `CandidateStore.search()` combines metadata filters (skills, languages, seniority, country,
   min years) with semantic similarity over chunks; with no query it is a pure field search.
4. **Agent** (`cv_screener/agent.py`). A PydanticAI agent with three tools: `search_candidates`,
   `get_candidate`, `list_filter_values`. It never receives the whole dataset; instructions require it to
   name only candidates returned by tools and to say when nobody matches.

## Evals and tests

- `evals/cases.yaml` has 8 cases (the four example questions, a field filter, and three no-match cases).
  Each checks tool usage, expected/forbidden names, and **grounding**: every candidate named in the answer
  must appear in tool results. Results are written to `evals/last_run.txt`.
- `tests/` runs offline with a hashing fake embedder, an in-memory Chroma, and PydanticAI's `TestModel`.

## Layout

```
cv_screener/  config, models, seeds, llm, generate, store, index, agent, cli, templates/
evals/        cases.yaml, run_evals.py
tests/        offline unit tests
data/         generated output (git-ignored)
```
