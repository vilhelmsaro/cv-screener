"""Vector store (Chroma) with structured metadata filters + semantic search."""
from __future__ import annotations

import re
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


def make_client() -> chromadb.ClientAPI:
    if settings.chroma_host:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def field_metadata(cid: str, f: ExtractedFields) -> dict:
    meta: dict = {
        "candidate_id": cid,
        "name": f.full_name,
        "title": f.current_title,
        "city": f.city,
        "country": norm(f.country),
        "seniority": f.seniority,
        "years_experience": f.years_experience,
        "skills": ", ".join(f.skills),
        "languages": ", ".join(f.languages),
        "education": f.highest_education,
        "summary": f.summary,
    }
    # One boolean flag per skill/language: exact, portable filtering in Chroma.
    meta.update({f"skill_{norm(s)}": True for s in f.skills})
    meta.update({f"lang_{norm(lang)}": True for lang in f.languages})
    return meta


def chunk_text(text: str, size: int = 900) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


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
        chunks = chunk_text(full_text)
        vectors = self.embedder.embed([profile_doc, *chunks])
        self.profiles.upsert(ids=[cid], documents=[full_text], embeddings=[vectors[0]], metadatas=[meta])
        self.chunks.delete(where={"candidate_id": cid})
        self.chunks.upsert(
            ids=[f"{cid}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors[1:],
            metadatas=[{**meta, "chunk": i} for i in range(len(chunks))],
        )

    # ---------- read ----------
    @staticmethod
    def build_where(skills=None, languages=None, seniority=None, country=None, min_years=None) -> dict | None:
        conds: list[dict] = [{f"skill_{norm(s)}": True} for s in skills or []]
        conds += [{f"lang_{norm(lang)}": True} for lang in languages or []]
        if seniority:
            conds.append({"seniority": {"$in": list(seniority)}})
        if country:
            conds.append({"country": norm(country)})
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

    def search(self, query: str | None = None, limit: int = 5, **filters) -> list[Hit]:
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
            if len(h.snippets) < 2:
                h.snippets.append(doc[:300])
        return sorted(hits.values(), key=lambda h: -(h.score or 0))[:limit]

    def all_profiles(self) -> list[dict]:
        return self.profiles.get(include=["metadatas"])["metadatas"]

    def get(self, name_or_id: str) -> dict | None:
        key = fold(name_or_id)
        res = self.profiles.get(include=["metadatas", "documents"])
        for cid, meta, doc in zip(res["ids"], res["metadatas"], res["documents"]):
            name = fold(meta["name"])
            if key == cid or key in name or all(part in name for part in key.split()):
                fields = {k: v for k, v in meta.items() if not k.startswith(("skill_", "lang_"))}
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
