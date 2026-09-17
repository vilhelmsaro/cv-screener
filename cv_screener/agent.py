"""Step 3: grounded chat agent that searches the index through tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from .llm import openrouter_model, settings
from .models import Seniority
from .store import CandidateStore

INSTRUCTIONS = """You are a recruiting assistant answering questions about a fixed dataset of candidate CVs.
Rules:
- You know nothing about candidates except what your tools return. Always call a tool before answering.
- Use search_candidates. Put hard requirements (skills, spoken languages, seniority, country, years)
  into filters and the soft intent into `query`. If a filter returns nothing, retry once with the term
  in `query` instead (the skill may be spelled differently), and call list_filter_values if unsure.
- For questions about one person, call get_candidate and base the answer on its CV text.
- Name every candidate you mention exactly as returned and say briefly why they match (cite the evidence).
- State only facts that appear in tool results. Search results list skills and languages as printed on
  the CV; for anything else (language level, employers, dates) call get_candidate.
- Never invent candidates, skills or facts. Semantic search always returns *something*: check that
  the snippets actually support the match before naming anyone.
- If nobody fits, say clearly that no candidate in the dataset matches.
- For "best fit" questions, rank the top 1-3 and explain the trade-offs."""


@dataclass
class Deps:
    store: CandidateStore


def build_agent(model: Model | str | None = None) -> Agent[Deps, str]:
    agent = Agent(model or openrouter_model(settings.agent_model), deps_type=Deps, instructions=INSTRUCTIONS)

    @agent.tool
    def search_candidates(
        ctx: RunContext[Deps],
        query: str | None = None,
        skills: list[str] | None = None,
        languages: list[str] | None = None,
        seniority: list[Seniority] | None = None,
        country: str | None = None,
        min_years: int | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search candidates by meaning (query) and/or exact fields.

        Args:
            query: free-text semantic query, e.g. "built recommender systems in production".
            skills: required technologies, all must match, e.g. ["Python"].
            languages: required spoken languages in English, e.g. ["Spanish"].
            seniority: allowed levels. Levels are ordered intern < junior < mid < senior < lead/staff,
                so a "senior" role also fits lead and staff.
            country: country name in English.
            min_years: minimum years of professional experience.
            limit: max candidates to return (capped at 20).
        """
        hits = ctx.deps.store.search(query=query, limit=min(limit, 20), skills=skills, languages=languages,
                                     seniority=seniority, country=country, min_years=min_years)
        return [asdict(h) for h in hits]

    @agent.tool
    def get_candidate(ctx: RunContext[Deps], name: str) -> dict:
        """Get one candidate's structured fields and full CV text by name (partial names work)."""
        return ctx.deps.store.get(name) or {"error": f"No candidate named '{name}' in the dataset."}

    @agent.tool
    def list_filter_values(ctx: RunContext[Deps]) -> dict:
        """List the skills, languages, countries and seniority values present in the index."""
        return ctx.deps.store.facet_values()

    return agent
