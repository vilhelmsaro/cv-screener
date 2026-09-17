from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from cv_screener.agent import Deps, build_agent
from evals.run_evals import check_case, mentioned

NAMES = ["Lucía Fernández Ortega", "Johannes Becker", "Anna Petrosyan"]


def test_agent_calls_tools_without_api_key(store):
    agent = build_agent(model=TestModel())  # TestModel calls every tool, no network
    result = agent.run_sync("Who speaks Spanish?", deps=Deps(store=store))
    called = {p.tool_name for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)}
    assert {"search_candidates", "get_candidate", "list_filter_values"} <= called


def _messages(tool_return: str, answer: str):
    return [
        ModelResponse(parts=[ToolCallPart(tool_name="search_candidates", args={})]),
        ModelRequest(parts=[ToolReturnPart(tool_name="search_candidates", content=tool_return)]),
        ModelResponse(parts=[TextPart(answer)]),
    ]


def test_eval_checker_flags_ungrounded_names():
    case = {"expect_all": ["Lucía Fernández Ortega"], "expect_tool": "search_candidates"}
    answer = "Lucía Fernández Ortega and Anna Petrosyan speak Spanish."
    checks = check_case(case, answer, _messages("[{'name': 'Lucía Fernández Ortega'}]", answer), NAMES)
    grounded = next(c for c in checks if c[0].startswith("grounded"))
    assert grounded[1] is False and "Anna Petrosyan" in grounded[2]


def test_eval_checker_no_match():
    case = {"expect_no_match": True}
    answer = "No candidate in the dataset speaks Japanese."
    assert all(c[1] for c in check_case(case, answer, _messages("[]", answer), NAMES))


def test_mentioned_matches_whole_words_only():
    names = ["Chen Wei", "Daniel Kim", "Lucía Fernández Ortega"]
    assert mentioned("The kitchen team skimmed weights.", names) == set()
    assert mentioned("Lucia Ortega and Daniel Kim fit.", names) == {"Lucía Fernández Ortega", "Daniel Kim"}


def test_model_never_receives_the_cv_dataset(store):
    """The core requirement of the task: the agent searches with tools instead of reading every CV."""
    sent = []

    def search_then_answer(messages, info):
        sent.append(repr(messages))
        if len(sent) == 1:
            return ModelResponse(parts=[ToolCallPart("search_candidates", {"languages": ["Spanish"]})])
        return ModelResponse(parts=[TextPart("Done.")])

    build_agent(FunctionModel(search_then_answer)).run_sync("Who speaks Spanish?", deps=Deps(store=store))
    for secret in ("Lucía", "Becker", "Petrosyan", "churn", "Firmware", "scikit-learn"):
        assert secret not in sent[0]
    # Not vacuous: candidate data does reach the model, but only after a tool returned it.
    assert "Lucía" in sent[1] and "Becker" not in sent[1]


def test_empty_search_explains_itself(store):
    """An empty list alone made the agent answer "nobody matches" instead of searching differently."""
    calls = []

    def call_search(messages, info):
        calls.append(1)
        if len(calls) == 1:  # a filter on wording no CV prints
            return ModelResponse(parts=[ToolCallPart("search_candidates", {"skills": ["Machine Learning"]})])
        return ModelResponse(parts=[TextPart("Done.")])

    with capture_run_messages() as messages:
        build_agent(FunctionModel(call_search)).run_sync("Best fit for a senior ML role?", deps=Deps(store=store))
    returned = next(p.content for m in messages for p in m.parts if isinstance(p, ToolReturnPart))
    assert returned["candidates"] == [] and "call list_filter_values" in returned["note"]

    hits = next(p.content for m in capture_returns(store) for p in m.parts if isinstance(p, ToolReturnPart))
    assert hits["candidates"] and "note" not in hits  # a search that finds people carries no note


def capture_returns(store):
    def search(messages, info):
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[ToolCallPart("search_candidates", {"languages": ["Spanish"]})])
        return ModelResponse(parts=[TextPart("Done.")])

    with capture_run_messages() as messages:
        build_agent(FunctionModel(search)).run_sync("Who speaks Spanish?", deps=Deps(store=store))
    return messages
