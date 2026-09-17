"""Step 2: read the PDFs, take structured fields from the page with rules, index into Chroma."""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import pymupdf
from rich.console import Console
from rich.table import Table

from .config import CVS_DIR, settings
from .countries import canonical_country
from .fields import extract_fields, find_location
from .llm import OpenRouterEmbedder, structured_agent
from .models import ExtractedFields
from .store import REQUIRED_SECTIONS, CandidateStore, chunk_cv, missing_sections

console = Console()

COUNTRY_INSTRUCTIONS = """You get the contact lines of a CV. Reply with only the country the person is based in,
as its English name. If the text does not say, reply with an empty string."""


def pdf_text(path) -> str:
    """PDF text with one blank line between layout blocks, so chunking can split on them.

    Plain get_text("text") has no blank lines, which would make each page a single chunk.
    """
    with pymupdf.open(path) as doc:
        blocks = [b[4].strip() for page in doc for b in page.get_text("blocks") if b[6] == 0]
    return "\n\n".join(b for b in blocks if any(ch.isalnum() for ch in b))  # drops lone bullet glyphs


def pdf_name(path) -> str:
    """The candidate's name: the largest text on the first page."""
    with pymupdf.open(path) as doc:
        spans = [span for block in doc[0].get_text("dict")["blocks"] if block["type"] == 0
                 for line in block["lines"] for span in line["spans"] if span["text"].strip()]
    if not spans:
        return ""
    top = max(span["size"] for span in spans)
    return " ".join(span["text"].strip() for span in spans if span["size"] >= top - 0.5)


def read_cv(path: Path, today: date | None = None) -> tuple[str, ExtractedFields, list[str]]:
    """PDF -> (text, fields from rules, fields the rules could not resolve). Offline and deterministic."""
    text, name = pdf_text(path), pdf_name(path)
    fields, unresolved = extract_fields(chunk_cv(text, name=name), name, today or date.today())
    return text, fields, unresolved


def _country_from_llm(text: str, name: str) -> str:
    """Last resort for a location the rules cannot read. The answer must be a real country or it is dropped."""
    chunks = chunk_cv(text, name=name)
    contact = find_location(chunks) or "\n".join(c.text for c in chunks if c.section in ("header", "contact"))
    agent = structured_agent(settings.extract_model, str, COUNTRY_INSTRUCTIONS, max_tokens=50)
    return canonical_country(agent.run_sync(contact[:1500]).output) or ""


def run() -> None:
    pdfs = sorted(CVS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs in data/cvs. Run `cvs generate` first.")
    store = CandidateStore(OpenRouterEmbedder())
    failed, sent = [], {}
    for pdf in pdfs:
        try:
            text, fields, unresolved = read_cv(pdf)
            if "country" in unresolved:
                fields.country = _country_from_llm(text, fields.full_name)
                console.print(f"  {pdf.stem}: country not readable by rules, "
                              f"LLM says {fields.country or 'nothing'!r}")
            sent[pdf.stem] = store.upsert(pdf.stem, fields, text)
        except Exception as e:  # keep indexing the others; upsert is idempotent, so a rerun is safe
            failed.append(pdf.stem)
            console.print(f"[red]failed {pdf.stem}:[/] {e!r}")
            continue
        console.print(f"indexed {pdf.stem}: {fields.full_name} | {fields.current_title} | {fields.country} | "
                      f"{fields.seniority}, {fields.years_experience}y | {', '.join(fields.languages)} | "
                      f"{len(fields.skills)} skills")
    for cid in sorted(set(store.section_counts()) - {pdf.stem for pdf in pdfs}):
        store.remove(cid)  # its PDF is gone from data/cvs, so it must not stay searchable
        console.print(f"removed {cid}: no PDF in {CVS_DIR.name}/ any more")
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
