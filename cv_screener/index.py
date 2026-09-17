"""Step 2: parse PDFs, extract structured fields with an LLM, index into Chroma."""
from __future__ import annotations

from collections import Counter

import pymupdf
from rich.console import Console
from rich.table import Table

from .config import CVS_DIR, settings
from .llm import OpenRouterEmbedder, structured_agent
from .models import ExtractedFields
from .store import REQUIRED_SECTIONS, CandidateStore, missing_sections

console = Console()

EXTRACT_INSTRUCTIONS = """Extract structured fields from the CV text. Use only what the CV states.
Seniority: intern/junior/mid/senior/lead/staff/manager based on title and years.
Spoken languages only (not programming languages), English names. Country in English."""


def pdf_text(path) -> str:
    """PDF text with one blank line between layout blocks, so chunking can split on them.

    Plain get_text("text") has no blank lines, which would make each page a single chunk.
    """
    with pymupdf.open(path) as doc:
        blocks = [b[4].strip() for page in doc for b in page.get_text("blocks") if b[6] == 0]
    return "\n\n".join(b for b in blocks if any(ch.isalnum() for ch in b))  # drops lone bullet glyphs


def run() -> None:
    pdfs = sorted(CVS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs in data/cvs. Run `cvs generate` first.")
    store = CandidateStore(OpenRouterEmbedder())
    extractor = structured_agent(settings.extract_model, ExtractedFields, EXTRACT_INSTRUCTIONS)
    failed, sent = [], {}
    for pdf in pdfs:
        try:
            text = pdf_text(pdf)
            fields = extractor.run_sync(text).output
            sent[pdf.stem] = store.upsert(pdf.stem, fields, text)
        except Exception as e:  # keep indexing the others; upsert is idempotent, so a rerun is safe
            failed.append(pdf.stem)
            console.print(f"[red]failed {pdf.stem}:[/] {e!r}")
            continue
        console.print(f"indexed {pdf.stem}: {fields.full_name} | {fields.seniority} | "
                      f"{', '.join(fields.languages)} | {len(fields.skills)} skills")
    console.print(f"[green]Done:[/] {store.profiles.count()} candidates, {store.chunks.count()} chunks")
    problems = check_coverage(store, sent)
    if failed or problems:
        raise SystemExit(f"Failed to index: {', '.join(failed) or 'none'}. Coverage problems: {len(problems)}.")


def check_coverage(store: CandidateStore, sent: dict[str, Counter] | None = None) -> list[str]:
    """Print chunks per section as read back from Chroma and return every problem found.

    Problems: a required section with no chunks, or (right after indexing) a count that differs from what
    was sent, which would mean chunks were lost on the way into the store.
    """
    stored = store.section_counts()
    if not stored and not sent:
        console.print("[red]coverage:[/] the store has no chunks. Run `cvs index` first.")
        return ["store is empty"]
    problems = []
    table = Table("id", "name", *REQUIRED_SECTIONS, "other", "total")
    for cid in sorted(set(stored) | set(sent or {})):
        counts = stored.get(cid, {}).get("sections", Counter())
        name = stored.get(cid, {}).get("name", "?")
        other = sum(n for s, n in counts.items() if s not in REQUIRED_SECTIONS)
        table.add_row(cid, name, *(str(counts[s]) for s in REQUIRED_SECTIONS), str(other), str(counts.total()))
        problems += [f"{cid}: no '{s}' chunks" for s in missing_sections(counts)]
        if sent is not None and sent.get(cid) != counts:
            problems.append(f"{cid}: sent {dict(sent.get(cid, {}))}, stored {dict(counts)}")
    console.print(table)
    for p in problems:
        console.print(f"[red]coverage:[/] {p}")
    if not problems:
        console.print("[green]Coverage OK:[/] every candidate has all required sections in the store.")
    return problems
