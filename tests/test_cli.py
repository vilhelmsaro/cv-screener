"""CLI behaviour: the tool trace, the empty-index guard and a clean exit from chat. No network."""
import uuid

import chromadb
import pytest
from typer.testing import CliRunner

from cv_screener import cli
from cv_screener.agent import Deps, build_agent
from cv_screener.store import CandidateStore

from .fakes import FakeEmbedder, scripted_model

runner = CliRunner()


@pytest.fixture
def spanish_agent(monkeypatch, store):
    model = scripted_model("search_candidates", {"languages": ["Spanish"]})
    monkeypatch.setattr(cli, "_agent_and_deps", lambda: (build_agent(model), Deps(store=store)))


def test_ask_prints_the_tool_trace_and_the_answer(spanish_agent):
    result = runner.invoke(cli.app, ["ask", "Which candidates speak Spanish?"])
    assert result.exit_code == 0
    assert "-> search_candidates(languages=['Spanish'])" in result.stderr  # what the answer is based on
    assert "<- 1 candidate(s): Lucía Fernández Ortega" in result.stderr
    assert "Lucía" in result.stdout


def test_ask_stops_on_an_empty_index(monkeypatch):
    real_store = CandidateStore
    monkeypatch.setattr("cv_screener.llm.OpenRouterEmbedder", FakeEmbedder)
    monkeypatch.setattr(  # a random prefix: EphemeralClients share state within a process
        "cv_screener.store.CandidateStore",
        lambda embedder: real_store(FakeEmbedder(), client=chromadb.EphemeralClient(),
                                    prefix=f"e{uuid.uuid4().hex[:8]}"))
    result = runner.invoke(cli.app, ["ask", "Who speaks Spanish?"])
    assert result.exit_code == 2 and "index is empty" in result.stdout


def test_chat_exits_cleanly_on_end_of_input(spanish_agent):
    result = runner.invoke(cli.app, ["chat"], input="")
    assert result.exit_code == 0 and result.exception is None
