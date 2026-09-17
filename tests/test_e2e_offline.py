"""index -> store -> agent tools -> CLI -> eval checks, on the real generated CVs, with no network.

Only the wiring is tested here: a fake embedder stands in for the real one and a scripted model for the
LLM, so this says nothing about answer quality. Judgment is what the live `cvs eval` run measures.
Skipped until `cvs generate` has produced data/cvs.
"""
import uuid

import chromadb
import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import TextPart
from typer.testing import CliRunner

from cv_screener import cli, index
from cv_screener.agent import Deps, build_agent
from cv_screener.config import CVS_DIR
from cv_screener.store import CandidateStore
from evals.run_evals import check_case

from .fakes import FakeEmbedder, scripted_model

PDFS = sorted(CVS_DIR.glob("*.pdf"))
pytestmark = pytest.mark.skipif(not PDFS, reason="no generated CVs; run `cvs generate` first")
runner = CliRunner()


@pytest.fixture(scope="module")
def indexed_store():
    """Runs `cvs index` over the real PDFs with a fake embedder and an in-memory Chroma."""
    store = CandidateStore(FakeEmbedder(), client=chromadb.EphemeralClient(),
                           prefix=f"e{uuid.uuid4().hex[:8]}", score_floor=0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(index, "OpenRouterEmbedder", FakeEmbedder)
        mp.setattr(index, "CandidateStore", lambda embedder: store)
        # A country the rules cannot read would call an LLM: fail loudly instead.
        mp.setattr(index, "_country_from_llm",
                   lambda *a: pytest.fail("the rules could not read the country"))
        result = runner.invoke(cli.app, ["index"])
    assert result.exit_code == 0, result.output  # index.run exits non-zero on failure or missing coverage
    assert store.profiles.count() == len(PDFS)
    return store


@pytest.mark.parametrize("args, case", [
    ({"languages": ["Spanish"]},
     {"expect_all": ["Lucía Fernández Ortega", "Mateo Rojas Quintero"], "expect_tool": "search_candidates"}),
    ({"languages": ["Japanese"]}, {"expect_no_match": True}),
])
def test_ask_answers_only_from_tool_results(monkeypatch, indexed_store, args, case):
    model = scripted_model("search_candidates", args)
    monkeypatch.setattr(cli, "_agent_and_deps", lambda: (build_agent(model), Deps(store=indexed_store)))
    with capture_run_messages() as messages:
        result = runner.invoke(cli.app, ["ask", "q"])

    assert result.exit_code == 0
    assert "-> search_candidates(" in result.stderr  # the trace shows what the answer rests on
    answer = next(p.content for m in reversed(messages) for p in m.parts if isinstance(p, TextPart))
    names = [m["name"] for m in indexed_store.all_profiles()]
    failed = [check for check in check_case(case, answer, messages, names) if not check[1]]
    assert not failed, f"{failed}\nanswer: {answer}"
