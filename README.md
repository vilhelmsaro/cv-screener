# CV Screener

Generates 12 synthetic candidates as PDF CVs with AI-generated photos, indexes them into a vector store
with structured fields, and answers questions about them through a CLI agent that searches with tools.

**Stack:** Python 3.12 · PydanticAI · OpenRouter (text, images, embeddings) · ChromaDB · WeasyPrint ·
PyMuPDF · Typer

## Quick start (Docker)

Needs Docker with Compose and an [OpenRouter](https://openrouter.ai/keys) key. A full run costs about
$1.20 with the models in `.env.example`; re-runs reuse the cache and cost nothing.

```bash
git clone https://github.com/vilhelmsaro/cv-screener.git && cd cv-screener
cp .env.example .env            # put your OPENROUTER_API_KEY in .env
docker compose build

docker compose run --rm app generate   # 12 candidates -> data/cvs/*.pdf
docker compose run --rm app index      # PDFs -> fields + embeddings -> Chroma
docker compose run --rm app chat       # ask questions
```

| Command | What it does | Needs a key |
|---|---|---|
| `generate` | Writes the CVs: profile JSON, photo, PDF. `--force` regenerates, `--only c01` picks one | yes |
| `index` | Reads the PDFs, extracts fields, writes Chroma, checks coverage | embeddings only |
| `coverage` | Re-checks what is stored, per CV section | no |
| `chat` | Interactive agent, keeps history | yes |
| `ask "question"` | One question and exit | yes |
| `eval` | Runs `evals/cases.yaml`, prints pass/fail and a total | yes |
| `--entrypoint pytest app` | The offline test suite | no |

`ask` and `chat` print every tool call and its result to stderr, so you can see what an answer is based on.

## Local run (without Docker)

Needs Python 3.11+ and the WeasyPrint system libraries
([install notes](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)). With `CHROMA_HOST`
empty, Chroma runs embedded in `data/chroma`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your key
cvs generate && cvs index && cvs chat
pytest
```

## How it works

**1. Generation** (`generate.py`, `seeds.py`). The seeds fix 12 different people: roles, levels,
countries, stacks, education and languages. An LLM writes each CV into a Pydantic-validated
`CandidateProfile`, an image model makes the headshot, and one of three HTML templates (`classic`,
`modern` with a sidebar, `compact` US Letter) is rendered to PDF. Before saving, the CV is read back with
the same rules indexing uses; if the level, years or languages contradict the seed, the model is told what
is wrong and rewrites it. Profiles and photos are cached.

**2. Indexing** (`index.py`, `fields.py`, `store.py`). Text comes from the **PDFs**, not from the
generator's JSON. Fields are read from the page with plain rules, so the same PDF always yields the same
fields: name from the largest text on page 1, skills and languages from their sections, title from the
first job, years from the summary ("8 years of experience") or else the job dates, country from the
contact line (checked against the ISO country list), seniority from title words or years. Only a country
the rules cannot read goes to the LLM, and its answer is kept only if it is a real country.

Two Chroma collections are written: `profiles` (one per candidate) and `chunks` (one per CV section and
one per job or degree, each tagged with its section and embedded with the candidate's name and title).
Fields travel as metadata, with normalized `skill_keys` / `lang_keys` for exact filters
(`{"skill_keys": {"$contains": "python"}}`). After writing, indexing reads back what Chroma stored and
fails if a CV lost chunks or is missing a required section; `cvs coverage` repeats that check for free.

**3. Search** (`store.py`). Exact metadata filters (skills, languages, seniority, country, minimum years)
combined with semantic similarity over chunks. With no query it is a pure field search. A semantic hit
must score close to the best hit, so weak matches are dropped instead of returned.

**4. Agent** (`agent.py`). A PydanticAI agent with three tools: `search_candidates`, `get_candidate`,
`list_filter_values`. It never receives the dataset, may name only candidates a tool returned, and says
plainly when nobody matches.

## Evals and tests

- **`evals/cases.yaml`, 11 cases:** the four example questions, a field filter, semantic-only and
  filter-plus-semantic search, a partial name, and three no-match cases. Each checks tool usage, expected
  and forbidden names, and **grounding**: every candidate named in the answer must appear in tool results.
  Output goes to `evals/last_run.txt`; the recorded runs are in `NOTES.md` (latest: 11/11).
- **`tests/`, 153 tests, offline and no API key**, using a hashing fake embedder, an in-memory Chroma and
  PydanticAI's `FunctionModel`. They cover the chunking rules and edge cases (`test_chunking.py`), each
  field rule (`test_fields.py`), CV layouts our templates never produce (`test_foreign_cvs.py`), a run
  through index → tools → CLI → eval checks (`test_e2e_offline.py`), and that the model never receives CV
  data (`test_agent_and_evals.py`). `test_generated_cvs.py` checks the real PDFs in `data/cvs` once
  `generate` has run: every job and degree is one chunk with all its bullets, and every extracted field
  matches the answer key. `test_boundaries.py` keeps the seeds and generated JSON off the read path, which
  a real deployment would not have.

## Layout

```
cv_screener/  config, models, seeds, llm, generate, store, fields, parsing, countries, index, agent, cli, templates/
evals/        cases.yaml, run_evals.py
tests/        offline tests
data/         generated output (git-ignored)
```
