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
    with pymupdf.open(path) as doc:
        return "\n\n".join(page.get_text("text") for page in doc).strip()


def run() -> None:
    pdfs = sorted(CVS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs in data/cvs. Run `cvs generate` first.")
    store = CandidateStore(OpenRouterEmbedder())
    extractor = structured_agent(settings.extract_model, ExtractedFields, EXTRACT_INSTRUCTIONS)
    for pdf in pdfs:
        text = pdf_text(pdf)
        fields = extractor.run_sync(text).output
        store.upsert(pdf.stem, fields, text)
        console.print(f"indexed {pdf.stem}: {fields.full_name} | {fields.seniority} | "
                      f"{', '.join(fields.languages)} | {len(fields.skills)} skills")
    console.print(f"[green]Done:[/] {store.profiles.count()} candidates, {store.chunks.count()} chunks")
