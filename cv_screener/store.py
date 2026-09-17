"""Vector store (Chroma) with structured metadata filters + semantic search."""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

import chromadb

from .config import CHROMA_DIR, settings
from .llm import Embedder
from .models import ExtractedFields

_ALIASES = {"c++": "cplusplus", "c#": "csharp", ".net": "dotnet", "node.js": "nodejs", "golang": "go"}


def norm(value: str) -> str:
    """Normalize a skill/language/country into a metadata-safe key."""
    v = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower().strip()
    v = _ALIASES.get(v, v)
    return re.sub(r"[^a-z0-9]+", "_", v).strip("_")


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
    if skill_keys := sorted({norm(s) for s in f.skills} - {""}):
        meta["skill_keys"] = skill_keys
    if lang_keys := sorted({norm(lang) for lang in f.languages} - {""}):
        meta["lang_keys"] = lang_keys
    return meta


SECTION_HEADINGS = {
    "profile", "summary", "professional summary", "about", "about me", "contact", "contacts",
    "experience", "professional experience", "work experience", "employment", "employment history",
    "education", "skills", "technical skills", "core skills", "languages", "projects", "publications",
    "certifications", "certificates", "courses", "awards", "volunteering", "interests", "hobbies",
}
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
# Start of a date range such as "Feb 2022 – Present", "2016 – 2019" or "03/2019 - 06/2022".
_DATE_RANGE = re.compile(rf"^(?:{_MONTH}\s+|\d{{1,2}}/)?\d{{4}}\s*[–—-]", re.IGNORECASE)


def _is_heading(block: str) -> bool:
    return block.strip().rstrip(":").casefold() in SECTION_HEADINGS


def _date_line_index(block: str) -> int | None:
    """Index of the first short line in the block that starts a date range, if any."""
    for i, line in enumerate(block.splitlines()[:4]):
        if len(line) <= 40 and _DATE_RANGE.match(line.strip()):
            return i
    return None


def chunk_cv(text: str, name: str = "", max_chars: int = 2000) -> list[str]:
    """Split CV text into meaningful chunks: one per section, and one per dated entry (job, degree).

    `text` is blank-line separated layout blocks (see index.pdf_text). A block that is a known heading
    starts a section; the candidate's name starts the header section, because two-column layouts put the
    sidebar first. Inside a section, an entry starts at the block holding its date range, or at the block
    before it when the dates start their own block (a wrapped title). Every chunk begins with its section
    heading, so a snippet shows where the evidence came from.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    sections: list[tuple[str, list[str]]] = [("", [])]
    for block in blocks:
        if _is_heading(block):
            sections.append((block.strip().rstrip(":").title(), []))
        elif name and fold(block.splitlines()[0]).strip() == fold(name).strip():
            sections.append(("", [block]))
        else:
            sections[-1][1].append(block)

    chunks: list[str] = []
    for heading, body in sections:
        entries: list[list[str]] = []
        for block in body:
            idx = _date_line_index(block)
            if idx == 0 and entries and len(entries[-1]) == 1 and _date_line_index(entries[-1][0]) is None \
                    and len(entries[-1][0]) <= 120:
                entries[-1].append(block)  # dates in their own block right after a wrapped title
            elif idx is not None or not entries:
                entries.append([block])
            else:
                entries[-1].append(block)
        for entry in entries:
            chunks.extend(_split_long("\n".join(entry), heading, max_chars))
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


@dataclass
class Hit:
    candidate_id: str
    name: str
    title: str
    location: str
    seniority: str
    years_experience: int
    score: float | None = None
    snippets: list[str] = field(default_factory=list)


class CandidateStore:
    def __init__(self, embedder: Embedder, client: chromadb.ClientAPI | None = None, prefix: str = "cv"):
        self.embedder = embedder
        self.client = client or make_client()
        # We pass embeddings ourselves, so no embedding function is needed.
        self.profiles = self.client.get_or_create_collection(f"{prefix}_profiles", embedding_function=None,
                                                             metadata={"hnsw:space": "cosine"})
        self.chunks = self.client.get_or_create_collection(f"{prefix}_chunks", embedding_function=None,
                                                           metadata={"hnsw:space": "cosine"})

    # ---------- write ----------
    def upsert(self, cid: str, fields: ExtractedFields, full_text: str) -> None:
        meta = field_metadata(cid, fields)
        profile_doc = f"{fields.full_name}. {fields.current_title}. {fields.summary} Skills: {meta['skills']}"
        chunks = chunk_cv(full_text, name=fields.full_name)
        # Prefix each chunk with who it belongs to so a lone bullet list still embeds in context.
        header = f"{fields.full_name}, {fields.current_title}\n"
        vectors = self.embedder.embed([profile_doc, *(header + c for c in chunks)])
        # Delete first: Chroma's upsert merges metadata keys, so a re-index could keep stale fields.
        self.profiles.delete(ids=[cid])
        self.profiles.add(ids=[cid], documents=[full_text], embeddings=[vectors[0]], metadatas=[meta])
        self.chunks.delete(where={"candidate_id": cid})
        self.chunks.add(
            ids=[f"{cid}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors[1:],
            metadatas=[{**meta, "chunk": i} for i in range(len(chunks))],
        )

    # ---------- read ----------
    @staticmethod
    def build_where(skills=None, languages=None, seniority=None, country=None, min_years=None) -> dict | None:
        conds: list[dict] = [{"skill_keys": {"$contains": norm(s)}} for s in skills or []]
        conds += [{"lang_keys": {"$contains": norm(lang)}} for lang in languages or []]
        if seniority:
            conds.append({"seniority": {"$in": list(seniority)}})
        if country:
            conds.append({"country_key": norm(country)})
        if min_years is not None:
            conds.append({"years_experience": {"$gte": int(min_years)}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    @staticmethod
    def _hit(meta: dict) -> Hit:
        return Hit(candidate_id=meta["candidate_id"], name=meta["name"], title=meta["title"],
                   location=f"{meta['city']}, {meta['country']}", seniority=meta["seniority"],
                   years_experience=meta["years_experience"])

    def search(self, query: str | None = None, limit: int = 10, **filters) -> list[Hit]:
        where = self.build_where(**filters)
        if not query:  # pure field search
            res = self.profiles.get(where=where, include=["metadatas"])
            return [self._hit(m) for m in res["metadatas"]][:limit]

        n = min(max(limit * 4, 10), self.chunks.count() or 1)
        res = self.chunks.query(query_embeddings=self.embedder.embed([query]), n_results=n, where=where,
                                include=["metadatas", "documents", "distances"])
        hits: dict[str, Hit] = {}
        for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
            h = hits.setdefault(meta["candidate_id"], self._hit(meta))
            h.score = max(h.score or 0.0, round(1 - dist, 3))
            if len(h.snippets) < 2:  # results are sorted by distance: best evidence first
                h.snippets.append(doc[:1200])
        return sorted(hits.values(), key=lambda h: -(h.score or 0))[:limit]

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
