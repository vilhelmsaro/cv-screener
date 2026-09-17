"""Vector store (Chroma) with structured metadata filters + semantic search."""
from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import chromadb

from .config import CHROMA_DIR, settings
from .countries import canonical_country
from .llm import Embedder
from .models import ExtractedFields

_ALIASES = {"c++": "cplusplus", "c#": "csharp", ".net": "dotnet", "node.js": "nodejs", "golang": "go"}


def norm(value: str) -> str:
    """Normalize a skill/language/country into a metadata-safe key."""
    v = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower().strip()
    v = _ALIASES.get(v, v)
    return re.sub(r"[^a-z0-9]+", "_", v).strip("_")


def skill_key_variants(skill: str) -> set[str]:
    """Filter keys for a skill as printed: "Java 17" also matches "Java", "Docker (basic)" matches "Docker",
    "HTML/CSS" matches "HTML" and "CSS"."""
    base = re.sub(r"\s*\(.*?\)", "", skill).strip()
    base = re.sub(r"\s+v?\d+(\.\d+)*$", "", base)
    keys = {norm(skill), norm(base)}
    if "/" in base and " " not in base:
        keys |= {norm(part) for part in base.split("/")}
    return keys


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def make_client(attempts: int = 15) -> chromadb.ClientAPI:
    if settings.chroma_host:
        # docker compose starts the app as soon as the Chroma container exists, not when it accepts
        # connections, and the Chroma image has no shell for a healthcheck. Retry instead.
        for attempt in range(attempts):
            try:
                return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            except ValueError:  # raised by HttpClient when the server is unreachable
                if attempt == attempts - 1:
                    raise
                time.sleep(1)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def field_metadata(cid: str, f: ExtractedFields) -> dict:
    meta: dict = {
        "candidate_id": cid,
        "name": f.full_name,
        "title": f.current_title,
        "city": f.city,
        "country": f.country,
        "country_key": norm(f.country),
        "seniority": f.seniority,
        "years_experience": f.years_experience,
        "skills": ", ".join(f.skills),
        "languages": ", ".join(f.languages),
        "education": f.highest_education,
        "summary": f.summary,
    }
    # Normalized list metadata for exact filtering with $contains. Chroma rejects empty lists.
    if skill_keys := sorted({k for s in f.skills for k in skill_key_variants(s)} - {""}):
        meta["skill_keys"] = skill_keys
    if lang_keys := sorted({norm(lang) for lang in f.languages} - {""}):
        meta["lang_keys"] = lang_keys
    return meta


# Canonical section -> headings that open it (compared case-insensitively, trailing colon ignored).
SECTIONS: dict[str, set[str]] = {
    "summary": {"profile", "summary", "professional summary", "about", "about me"},
    "contact": {"contact", "contacts", "contact information"},
    "experience": {"experience", "professional experience", "work experience", "work history", "employment",
                   "employment history"},
    "education": {"education"},
    "skills": {"skills", "technical skills", "core skills", "key skills"},
    "languages": {"languages"},
    "projects": {"projects", "selected projects", "personal projects"},
    "publications": {"publications"},
    "certifications": {"certifications", "certificates", "courses"},
    "awards": {"awards"},
    "volunteering": {"volunteering"},
    "interests": {"interests", "hobbies"},
}
_HEADING_TO_SECTION = {heading: key for key, headings in SECTIONS.items() for heading in headings}
HEADER = "header"  # name, headline, contact line and any text before the first heading
REQUIRED_SECTIONS = ("header", "experience", "education", "skills", "languages")

MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
# Start of a date range: "Feb 2022 – Present", "2016 – 2019", "03/2019 - 06/2022", "01.2020 – 05.2022".
_DATE_RANGE = re.compile(rf"^(?:{MONTH}\s+|\d{{1,2}}[./])?\d{{4}}\s*[–—-]", re.IGNORECASE)
# The same range at the end of a title line: "Software Engineer | 10Web | Yerevan 2023 - 2025".
_END = rf"(?:{MONTH}\s+|\d{{1,2}}[./])?\d{{4}}\s*[–—-]\s*(?:(?:{MONTH}\s+|\d{{1,2}}[./])?\d{{4}}|present|current|now)"
_DATE_TAIL = re.compile(rf"{_END}\s*[).\]]?\s*$", re.IGNORECASE)
_LABEL_LINE = re.compile(r"^([A-Za-z][\w &/-]{0,30}):\s*(\S.*)$")


class Chunk(NamedTuple):
    section: str  # a key of SECTIONS, or HEADER
    text: str


def _section_of(block: str) -> str | None:
    return _HEADING_TO_SECTION.get(" ".join(block.split()).rstrip(":").casefold())


def _is_name_line(block: str, name: str) -> bool:
    """First line of the block is the candidate's name, allowing accents and an extra middle name."""
    line_words = re.findall(r"\w+", fold(block.splitlines()[0]))
    name_words = re.findall(r"\w+", fold(name))
    return bool(name_words) and set(name_words) <= set(line_words) and len(line_words) <= len(name_words) + 2


def date_line_index(block: str) -> int | None:
    """Index of the first line in the block that carries an entry's dates, if any."""
    for i, line in enumerate(block.splitlines()[:4]):
        if is_entry_header(line):
            return i
    return None


def is_entry_header(line: str) -> bool:
    """A date line of its own ("Feb 2022 - Present") or a title line ending in its dates.

    Length limits keep a bullet that merely starts or ends with years from opening an entry.
    """
    line = line.strip()
    return bool(len(line) <= 40 and _DATE_RANGE.match(line)) or bool(len(line) <= 120 and _DATE_TAIL.search(line))


def _inline_section_label(line: str, current: str) -> tuple[str | None, str]:
    """"Languages: Armenian (Native) | English" -> ("languages", "Armenian (Native) | English").

    Real CVs often label a short section instead of giving it a heading. Inside a skills section the same
    shape is a skill group ("Languages: Python, SQL"), so it is left alone there.
    """
    if m := _LABEL_LINE.match(line.strip()):
        key = _section_of(m.group(1))
        if key and key != current and not (current == "skills" and key == "languages"):
            return key, m.group(2).strip()
    return None, ""


def _split_at_boundaries(blocks: list[str]) -> list[str]:
    """Cut blocks where a heading, a labelled section or an entry header sits inside them.

    PDF layout blocks follow the page, not the document: a real CV often ends a paragraph block with the
    next section's heading, or runs one job's bullets into the next job's title line.
    """
    out: list[str] = []
    current = None  # the section we are in, carried across blocks
    printed: dict[str, str] = {}  # section key -> heading as printed, to reopen it after a labelled line
    for block in blocks:
        buffer: list[str] = []
        for line in block.splitlines():
            key = _section_of(line)
            label_key, rest = _inline_section_label(line, current or "")
            if key:
                if buffer:
                    out.append("\n".join(buffer))
                out.append(line.strip())
                buffer, current = [], key
                printed[key] = line.strip()
            elif label_key:
                if buffer:
                    out.append("\n".join(buffer))
                label = m.group(1).strip() if (m := _LABEL_LINE.match(line.strip())) else label_key
                # A labelled section is exactly one line long, so reopen the section it interrupted.
                out += [label, rest] + ([heading] if (heading := printed.get(current)) else [])
                buffer = []
            elif buffer and _is_title_with_dates(line):
                out.append("\n".join(buffer))  # "Software Engineer | 10Web | Yerevan 2023 - 2025"
                buffer = [line]
            elif len(buffer) > 1 and is_entry_header(line) and _title_like(buffer[-1]):
                out.append("\n".join(buffer[:-1]))  # the dates belong to the title line above them
                buffer = [buffer[-1], line]
            else:
                buffer.append(line)
        if buffer:
            out.append("\n".join(buffer))
    return out


def _is_title_with_dates(line: str) -> bool:
    """A title line that ends with its dates. A line that is only dates belongs to the title above it."""
    line = line.strip()
    return len(line) <= 120 and bool(_DATE_TAIL.search(line)) and not _DATE_RANGE.match(line)


def _title_like(line: str) -> bool:
    return bool(line.strip()) and len(line.strip()) <= 120 and not line.strip().endswith(".")


def chunk_cv(text: str, name: str = "", max_chars: int = 2000) -> list[Chunk]:
    """Split CV text into meaningful chunks: one per section, and one per dated entry (job, degree).

    `text` is blank-line separated layout blocks (see index.pdf_text). A block that is a known heading
    starts a section; the candidate's name starts the header section, because two-column layouts put the
    sidebar first. Inside a section, an entry starts at the block holding its date range, or at the block
    before it when the dates start their own block (a wrapped title). Every chunk text begins with the
    heading as printed, so a snippet shows where the evidence came from. No block is dropped.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    blocks = [b for b in _split_at_boundaries(blocks) if b.strip()]
    sections: list[tuple[str, str, list[str]]] = [(HEADER, "", [])]  # (key, printed heading, blocks)
    for block in blocks:
        if key := _section_of(block):
            sections.append((key, " ".join(block.split()).rstrip(":").title(), []))
        elif name and _is_name_line(block, name):
            sections.append((HEADER, "", [block]))
        else:
            sections[-1][2].append(block)

    chunks: list[Chunk] = []
    for key, heading, body in sections:
        entries: list[list[str]] = []
        for block in body:
            idx = date_line_index(block)
            if idx == 0 and entries and len(entries[-1]) == 1 and date_line_index(entries[-1][0]) is None \
                    and len(entries[-1][0]) <= 120:
                entries[-1].append(block)  # dates in their own block right after a wrapped title
            elif idx is not None or not entries:
                entries.append([block])
            else:
                entries[-1].append(block)
        for entry in entries:
            chunks.extend(Chunk(key, t) for t in _split_long("\n".join(entry), heading, max_chars))
    return chunks


def _split_long(entry: str, heading: str, max_chars: int) -> list[str]:
    prefix = f"{heading}\n" if heading else ""
    if len(entry) <= max_chars:
        return [prefix + entry]
    # Rare: keep the entry's first line (e.g. the job title) on every part.
    lines = entry.splitlines()
    parts, cur = [], lines[0]
    for line in lines[1:]:
        if len(cur) + len(line) > max_chars:
            parts.append(cur)
            cur = lines[0]
        cur += "\n" + line
    parts.append(cur)
    return [prefix + p for p in parts]


def missing_sections(section_counts: dict[str, int]) -> list[str]:
    return [s for s in REQUIRED_SECTIONS if not section_counts.get(s)]


@dataclass
class Hit:
    candidate_id: str
    name: str
    title: str
    location: str
    seniority: str
    years_experience: int
    skills: str = ""      # as printed on the CV, so the agent can cite a match without another call
    languages: str = ""
    score: float | None = None
    snippets: list[str] = field(default_factory=list)


class CandidateStore:
    # A chunk counts as a match only near the best one and never below the floor: cosine similarity ranks
    # everything, so without this a query returns its least-bad chunks as if they were matches. Measured
    # with text-embedding-3-small on this dataset: real matches 0.44-0.57, unrelated CVs <= 0.34, a
    # nonsense query <= 0.22. The scale belongs to the embedding model, so tests with a fake embedder
    # pass score_floor=0.
    def __init__(self, embedder: Embedder | None, client: chromadb.ClientAPI | None = None, prefix: str = "cv",
                 score_floor: float = 0.25, score_ratio: float = 0.7):
        self.embedder = embedder  # None is fine for reads that need no query vector (e.g. `cvs coverage`)
        self.score_floor, self.score_ratio = score_floor, score_ratio
        self.client = client or make_client()
        # We pass embeddings ourselves, so no embedding function is needed.
        self.profiles = self.client.get_or_create_collection(f"{prefix}_profiles", embedding_function=None,
                                                             metadata={"hnsw:space": "cosine"})
        self.chunks = self.client.get_or_create_collection(f"{prefix}_chunks", embedding_function=None,
                                                           metadata={"hnsw:space": "cosine"})

    # ---------- write ----------
    def upsert(self, cid: str, fields: ExtractedFields, full_text: str) -> Counter[str]:
        """Write one candidate. Returns how many chunks were sent per section, for the coverage check."""
        meta = field_metadata(cid, fields)
        profile_doc = f"{fields.full_name}. {fields.current_title}. {fields.summary} Skills: {meta['skills']}"
        chunks = chunk_cv(full_text, name=fields.full_name)
        # Prefix each chunk with who it belongs to so a lone bullet list still embeds in context.
        header = f"{fields.full_name}, {fields.current_title}\n"
        vectors = self.embedder.embed([profile_doc, *(header + c.text for c in chunks)])
        # Delete first: Chroma's upsert merges metadata keys, so a re-index could keep stale fields.
        self.profiles.delete(ids=[cid])
        self.profiles.add(ids=[cid], documents=[full_text], embeddings=[vectors[0]], metadatas=[meta])
        self.chunks.delete(where={"candidate_id": cid})
        self.chunks.add(
            ids=[f"{cid}-{i}" for i in range(len(chunks))],
            documents=[c.text for c in chunks],
            embeddings=vectors[1:],
            metadatas=[{**meta, "chunk": i, "section": c.section} for i, c in enumerate(chunks)],
        )
        return Counter(c.section for c in chunks)

    def remove(self, cid: str) -> None:
        """Forget a candidate completely: used when their PDF is gone from data/cvs."""
        self.profiles.delete(ids=[cid])
        self.chunks.delete(where={"candidate_id": cid})

    def section_counts(self) -> dict[str, dict]:
        """Chunks per section for every candidate, read back from Chroma: what search can actually see."""
        out: dict[str, dict] = {}
        for m in self.chunks.get(include=["metadatas"])["metadatas"]:
            entry = out.setdefault(m["candidate_id"], {"name": m["name"], "sections": Counter()})
            entry["sections"][m.get("section", "unknown")] += 1
        return out

    # ---------- read ----------
    @staticmethod
    def build_where(skills=None, languages=None, seniority=None, country=None, min_years=None) -> dict | None:
        conds: list[dict] = [{"skill_keys": {"$contains": norm(s)}} for s in skills or []]
        conds += [{"lang_keys": {"$contains": norm(lang)}} for lang in languages or []]
        if seniority:
            conds.append({"seniority": {"$in": list(seniority)}})
        if country:
            conds.append({"country_key": norm(canonical_country(country) or country)})
        if min_years is not None:
            conds.append({"years_experience": {"$gte": int(min_years)}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    @staticmethod
    def _hit(meta: dict) -> Hit:
        return Hit(candidate_id=meta["candidate_id"], name=meta["name"], title=meta["title"],
                   location=", ".join(dict.fromkeys(p for p in (meta["city"], meta["country"]) if p)),
                   seniority=meta["seniority"], years_experience=meta["years_experience"],
                   skills=meta["skills"], languages=meta["languages"])

    def search(self, query: str | None = None, limit: int = 10, **filters) -> list[Hit]:
        where = self.build_where(**filters)
        if not query:  # pure field search
            res = self.profiles.get(where=where, include=["metadatas"])
            return [self._hit(m) for m in res["metadatas"]][:limit]

        if self.embedder is None:
            raise RuntimeError("Semantic search needs an embedder.")
        n = min(max(limit * 4, 10), self.chunks.count() or 1)
        res = self.chunks.query(query_embeddings=self.embedder.embed([query]), n_results=n, where=where,
                                include=["metadatas", "documents", "distances"])
        hits: dict[str, Hit] = {}
        for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
            h = hits.setdefault(meta["candidate_id"], self._hit(meta))
            h.score = max(h.score or 0.0, round(1 - dist, 3))
            if len(h.snippets) < 2:  # results are sorted by distance: best evidence first
                h.snippets.append(doc[:1200])
        ranked = sorted(hits.values(), key=lambda h: -(h.score or 0))
        if not ranked:
            return []
        cutoff = max(self.score_floor, self.score_ratio * (ranked[0].score or 0))
        return [h for h in ranked if (h.score or 0) >= cutoff][:limit]

    def all_profiles(self) -> list[dict]:
        return self.profiles.get(include=["metadatas"])["metadatas"]

    def get(self, name_or_id: str) -> dict | None:
        key = fold(name_or_id).strip()
        if not key:
            return None
        res = self.profiles.get(include=["metadatas", "documents"])
        for cid, meta, doc in zip(res["ids"], res["metadatas"], res["documents"]):
            # Whole-word match, any order: "lucia fernandez" finds "Lucía Fernández Ortega", "an" finds nobody.
            if key == cid or set(key.split()) <= set(fold(meta["name"]).split()):
                fields = {k: v for k, v in meta.items() if not k.endswith("_keys") and k != "country_key"}
                return {"fields": fields, "cv_text": doc}
        return None

    def facet_values(self) -> dict[str, list[str]]:
        metas = self.all_profiles()
        out: dict[str, set] = {"skills": set(), "languages": set(), "countries": set(), "seniority": set()}
        for m in metas:
            out["skills"].update(s.strip() for s in m["skills"].split(",") if s.strip())
            out["languages"].update(s.strip() for s in m["languages"].split(",") if s.strip())
            out["countries"].add(m["country"])
            out["seniority"].add(m["seniority"])
        return {k: sorted(v) for k, v in out.items()}
