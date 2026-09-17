"""Offline stand-ins shared by tests."""
import hashlib
import math
import re

from cv_screener.models import ExtractedFields


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no network, no key."""

    dim = 256

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9+#]+", t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def make_fields(name, title, country, seniority, years, skills, languages):
    return ExtractedFields(full_name=name, current_title=title, city="X", country=country, seniority=seniority,
                           years_experience=years, skills=skills, languages=languages,
                           highest_education="MSc", summary=f"{title} with {years} years.")


def scripted_model(tool: str, args: dict):
    """Calls one tool, then answers naming exactly the candidates that tool returned.

    Only for search_candidates: the other tools return a dict, not a list of hits.
    """
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info) -> ModelResponse:
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart(tool, args)])
        names = [hit["name"] for hit in returns[-1].content]
        answer = f"{', '.join(names)} match." if names else "No candidate in the dataset matches."
        return ModelResponse(parts=[TextPart(answer)])

    return FunctionModel(respond)
