"""Generation checks that need no network: seed consistency and the retry on a wrong CV."""
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cv_screener import generate
from cv_screener.fields import parse_languages, seniority_for
from cv_screener.generate import seed_mismatches
from cv_screener.models import CandidateProfile, Seed
from cv_screener.seeds import SEEDS, SEEDS_BY_ID

SEED = SEEDS_BY_ID["c11"]  # mid, 6 years, Romanian/English/Italian


def profile(title="QA Automation Engineer", summary="QA Automation Engineer with 6 years of experience.",
            languages=("Romanian", "English", "Italian")) -> CandidateProfile:
    return CandidateProfile(
        full_name=SEED.name, headline=title, email="e@example.com", phone="+40 700 000 000",
        location="Bucharest, Romania", links=[], summary=summary,
        experience=[{"title": title, "company": "Bitdefender", "location": "Bucharest", "start": "Feb 2023",
                     "end": "Present", "bullets": ["Built a Playwright suite."]}],
        education=[], skills=[], languages=[{"name": n, "level": "C1"} for n in languages],
    )


def test_consistent_profile_has_no_mismatches():
    assert seed_mismatches(SEED, profile()) == []


@pytest.mark.parametrize("kwargs, fragment", [
    ({"title": "Senior QA Automation Engineer"}, "reads as senior, but the person is mid"),
    ({"summary": "QA engineer with 8 years of experience."}, '"6 years of experience" (found: 8)'),
    ({"summary": "Experienced QA engineer."}, "(found: None)"),
    ({"languages": ("Romanian", "English")}, "spoken languages must be exactly"),
])
def test_each_mismatch_is_reported(kwargs, fragment):
    problems = seed_mismatches(SEED, profile(**kwargs))
    assert len(problems) == 1 and fragment in problems[0]


def test_seeds_are_unique_and_templates_exist():
    assert len(SEEDS) >= 10  # the task asks for at least 10 candidates
    assert len(SEEDS_BY_ID) == len(SEEDS) and len({s.name for s in SEEDS}) == len(SEEDS)
    for seed in SEEDS:
        assert (Path(generate.__file__).parent / "templates" / f"{seed.template}.html").exists(), seed.id


def test_invalid_seed_is_rejected():
    fields = SEED.model_dump()
    with pytest.raises(ValidationError):
        Seed(**{**fields, "level": "mdi"})
    with pytest.raises(ValidationError):
        Seed(**{**fields, "template": "fancy"})
    with pytest.raises(ValidationError):
        Seed(**fields, stak="typo in a key")


def test_every_seed_can_be_satisfied():
    # A CV that prints the seed's own role must pass, otherwise generation could never succeed.
    for seed in SEEDS:
        assert seniority_for(seed.role, seed.years) == seed.level, seed.id
        assert parse_languages([seed.languages]), seed.id


def test_wrong_cv_is_sent_back_and_fixed(monkeypatch):
    replies = [profile(title="Senior QA Automation Engineer"), profile()]
    retry_messages = []

    def fake_llm(messages, info: AgentInfo) -> ModelResponse:
        retry_messages.extend(p.content for m in messages for p in m.parts if isinstance(p, RetryPromptPart))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, replies.pop(0).model_dump())])

    monkeypatch.setattr(generate, "structured_agent",
                        lambda *a, **k: Agent(FunctionModel(fake_llm), output_type=CandidateProfile, retries=2))
    result = generate._profile(SEED)
    assert result.experience[0].title == "QA Automation Engineer"
    assert any("reads as senior" in str(m) for m in retry_messages)


def test_a_skill_item_with_a_comma_becomes_two_skills():
    # The PDF renders items comma-separated, so "Git, Jenkins" as one item could never be read back as one.
    from cv_screener.models import SkillGroup
    assert SkillGroup(group="Tools", items=["Git, Jenkins", "Docker"]).items == ["Git", "Jenkins", "Docker"]
    # Commas inside brackets are part of the skill.
    assert SkillGroup(group="Cloud", items=["AWS (EC2, S3)"]).items == ["AWS (EC2, S3)"]
