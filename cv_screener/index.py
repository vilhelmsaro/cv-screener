"""Step 2: parse PDFs, extract structured fields with an LLM, index into Chroma."""
from __future__ import annotations

import pymupdf
from rich.console import Console

from .config import CVS_DIR, settings
from .llm import OpenRouterEmbedder, structured_agent
from .models import ExtractedFields
from .store import CandidateStore

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
    failed = []
    for pdf in pdfs:
        try:
            text = pdf_text(pdf)
            fields = extractor.run_sync(text).output
            store.upsert(pdf.stem, fields, text)
        except Exception as e:  # keep indexing the others; upsert is idempotent, so a rerun is safe
            failed.append(pdf.stem)
            console.print(f"[red]failed {pdf.stem}:[/] {e!r}")
            continue
        console.print(f"indexed {pdf.stem}: {fields.full_name} | {fields.seniority} | "
                      f"{', '.join(fields.languages)} | {len(fields.skills)} skills")
    console.print(f"[green]Done:[/] {store.profiles.count()} candidates, {store.chunks.count()} chunks")
    if failed:
        raise SystemExit(f"Failed to index: {', '.join(failed)}. Rerun `cvs index`.")
