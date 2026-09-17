"""CLI entry point: `cvs generate | index | coverage | chat | ask | eval`."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def generate(force: bool = typer.Option(False, help="Regenerate cached profiles/photos"),
             only: list[str] | None = typer.Option(None, help="Seed ids, e.g. --only c01")):
    """Generate synthetic candidates: profile JSON, AI photo, PDF CV."""
    from . import generate as gen
    gen.run(force=force, only=only)


@app.command()
def index():
    """Parse PDFs, extract fields, write to the vector store."""
    from . import index as idx
    idx.run()


@app.command()
def coverage():
    """Check what is in the vector store: chunks per CV section for every candidate. No API key needed."""
    from .index import check_coverage
    from .store import CandidateStore
    raise typer.Exit(1 if check_coverage(CandidateStore(embedder=None)) else 0)


def _agent_and_deps():
    from .agent import Deps, build_agent
    from .llm import OpenRouterEmbedder
    from .store import CandidateStore
    return build_agent(), Deps(store=CandidateStore(OpenRouterEmbedder()))


@app.command()
def ask(question: str):
    """Ask one question and exit."""
    agent, deps = _agent_and_deps()
    console.print(Markdown(agent.run_sync(question, deps=deps).output))


@app.command()
def chat():
    """Interactive chat over the dataset. Empty line or 'exit' to quit."""
    agent, deps = _agent_and_deps()
    history = []
    console.print("[bold]CV Screener chat[/] - ask about the candidates.")
    while True:
        q = console.input("[cyan]you> [/]").strip()
        if q.lower() in {"", "exit", "quit"}:
            break
        try:
            result = agent.run_sync(q, deps=deps, message_history=history)
        except Exception as e:  # a failed API call should not end the session; history stays as it was
            console.print(f"[red]Error:[/] {e!r}")
            continue
        history = result.all_messages()
        console.print(Markdown(result.output))


@app.command(name="eval")
def run_eval():
    """Run the eval suite and print pass/fail."""
    import sys
    from .config import ROOT
    sys.path.insert(0, str(ROOT))
    from evals.run_evals import main
    raise typer.Exit(main())


if __name__ == "__main__":
    app()
