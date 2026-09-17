# CLAUDE.md

Project: CV Screener test task. Python only. See README.md for architecture.

## Commands
- `pytest` — offline tests, must pass without OPENROUTER_API_KEY
- `cvs generate | index | chat | ask "..." | eval`
- Docker: `docker compose run --rm app <command>`

## Conventions
- All LLM calls go through `cv_screener/llm.py` (OpenRouter). Structured outputs use Pydantic models in `models.py`.
- The agent must only see data via tools in `agent.py`; never put all CVs into the prompt.
- Keep `seeds.py` names stable: evals in `evals/cases.yaml` depend on them.
- Chroma metadata must be scalar; skills/languages are stored as `skill_<norm>` / `lang_<norm>` boolean flags.
- Keep `chromadb` pin in pyproject.toml equal to the image tag in docker-compose.yml.
- Never commit `.env` or anything under `data/` except `.gitkeep`.
- Commit messages in English, imperative mood, one logical change per commit.
