# CLAUDE.md

Project: CV Screener test task. Python only. See README.md for architecture.

## Commands
- `pytest` — offline tests, must pass without OPENROUTER_API_KEY
- `cvs generate | index | coverage | chat | ask "..." | eval`
- Docker: `docker compose run --rm app <command>`

## Conventions
- All LLM calls go through `cv_screener/llm.py` (OpenRouter). Structured outputs use Pydantic models in `models.py`.
- The agent must only see data via tools in `agent.py`; never put all CVs into the prompt.
- Keep `seeds.py` names stable: evals in `evals/cases.yaml` depend on them.
- Skills/languages are stored as normalized list metadata (`skill_keys`, `lang_keys`) and filtered with `$contains`;
  Chroma rejects empty lists, so omit the key instead.
- Indexing reads the PDFs. `seeds.py` and `data/profiles` are generator input and test answer keys only; the
  index/search/agent path must never import them (`tests/test_boundaries.py` enforces this).
- Keep `chromadb` pin in pyproject.toml equal to the image tag in docker-compose.yml.
- Never commit `.env` or anything under `data/` except `.gitkeep`. Real CVs live in `examples/`, which is
  committed and used as a parsing fixture by `tests/test_example_cv.py`.
- Commit messages in English, imperative mood, one logical change per commit.
