import pymupdf
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from cv_screener.agent import Deps, build_agent
from cv_screener.index import pdf_text
from evals.run_evals import check_case

NAMES = ["Lucía Fernández Ortega", "Johannes Becker", "Anna Petrosyan"]


def test_pdf_text_extraction(tmp_path):
    path = tmp_path / "cv.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Johannes Becker\nSenior Embedded Engineer\nC++, RTOS")
    doc.save(path)
    text = pdf_text(path)
    assert "Johannes Becker" in text and "RTOS" in text


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
