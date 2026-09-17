"""Thin wrappers around OpenRouter: structured-output agents, embeddings, image generation."""
from __future__ import annotations

import base64
from typing import Protocol

import httpx
from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .config import OPENROUTER_BASE_URL, settings


def openrouter_model(name: str) -> OpenRouterModel:
    return OpenRouterModel(name, provider=OpenRouterProvider(api_key=settings.require_key()))


def structured_agent(model_name: str, output_type: type, instructions: str, max_tokens: int = 4000) -> Agent:
    # Without max_tokens OpenRouter reserves the model's full output limit (65K for some) against credits.
    return Agent(openrouter_model(model_name), output_type=output_type, instructions=instructions, retries=2,
                 model_settings={"max_tokens": max_tokens})


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenRouterEmbedder:
    def __init__(self, model: str | None = None):
        self.model = model or settings.embed_model
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.require_key())

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def generate_image(prompt: str, model: str | None = None) -> bytes:
    """Generate one image via OpenRouter chat completions with image output."""
    r = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {settings.require_key()}"},
        json={
            "model": model or settings.image_model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"],
        },
        timeout=120,
    )
    r.raise_for_status()
    message = r.json()["choices"][0]["message"]
    images = message.get("images") or []
    if not images:
        raise RuntimeError(f"No image returned: {str(message)[:200]}")
    data_url: str = images[0]["image_url"]["url"]
    return base64.b64decode(data_url.split(",", 1)[1])
