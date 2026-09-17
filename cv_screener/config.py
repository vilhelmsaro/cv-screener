"""Settings loaded from environment (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROFILES_DIR = DATA / "profiles"  # generated source JSON (ground truth for generation only)
PHOTOS_DIR = DATA / "photos"
CVS_DIR = DATA / "cvs"
CHROMA_DIR = DATA / "chroma"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    api_key: str | None = os.getenv("OPENROUTER_API_KEY") or None
    gen_model: str = os.getenv("GEN_MODEL", "google/gemini-2.5-flash")
    extract_model: str = os.getenv("EXTRACT_MODEL", "google/gemini-2.5-flash")
    agent_model: str = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")
    image_model: str = os.getenv("IMAGE_MODEL", "google/gemini-2.5-flash-image")
    embed_model: str = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")
    chroma_host: str | None = os.getenv("CHROMA_HOST") or None
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))

    def require_key(self) -> str:
        if not self.api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return self.api_key


settings = Settings()
