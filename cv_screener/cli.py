"""CLI entry point: `cvs generate | index | coverage | chat | ask | eval`."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
err_console = Console(stderr=True)  # tool trace: visible in the terminal, kept out of a piped answer


def _print_trace(messages) -> None:
    """One dim line per tool call and per result, so it is visible what the answer is based on."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                args = ", ".join(f"{k}={v!r}" for k, v in part.args_as_dict().items())
                line = f"-> {part.tool_name}({args})"
            elif isinstance(part, ToolReturnPart):
                content = part.content
                if isinstance(content, dict) and "candidates" in content:
                    names = ", ".join(h["name"] for h in content["candidates"])
                    line = f"<- {len(content['candidates'])} candidate(s)" + (f": {names}" if names else "")
                elif isinstance(content, dict) and "error" in content:
                    line = f"<- {content['error']}"
                elif isinstance(content, dict) and "fields" in content:
                    line = f"<- CV of {content['fields']['name']}"
                else:
                    line = "<- filter values"
            else:
                continue
            # markup=False: args like ['Spanish'] would be read as Rich markup; soft_wrap: no hard wrapping.
            err_console.print(line, style="dim", markup=False, soft_wrap=True)


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
    store = CandidateStore(OpenRouterEmbedder())
    if store.profiles.count() == 0:  # otherwise every question looks like a genuine "nobody matches"
        console.print("The index is empty. Run `cvs generate` and `cvs index` first.")
        raise typer.Exit(2)
    return build_agent(), Deps(store=store)


@app.command()
def ask(question: str):
    """Ask one question and exit."""
    agent, deps = _agent_and_deps()
    result = agent.run_sync(question, deps=deps)
    _print_trace(result.new_messages())
    console.print(Markdown(result.output))


@app.command()
def chat():
    """Interactive chat over the dataset. Empty line or 'exit' to quit."""
    agent, deps = _agent_and_deps()
    history = []
    console.print("[bold]CV Screener chat[/] - ask about the candidates.")
    while True:
        try:
            q = console.input("[cyan]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C end the session, not with a traceback
            break
        if q.lower() in {"", "exit", "quit"}:
            break
        try:
            result = agent.run_sync(q, deps=deps, message_history=history)
        except Exception as e:  # a failed API call should not end the session; history stays as it was
            console.print(f"[red]Error:[/] {e!r}")
            continue
        history = result.all_messages()
        _print_trace(result.new_messages())
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
