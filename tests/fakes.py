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
