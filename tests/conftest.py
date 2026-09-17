import hashlib
import math
import re
import uuid

import chromadb
import pytest

from cv_screener.models import ExtractedFields
from cv_screener.store import CandidateStore


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


def _fields(name, title, country, seniority, years, skills, languages):
    return ExtractedFields(full_name=name, current_title=title, city="X", country=country, seniority=seniority,
                           years_experience=years, skills=skills, languages=languages,
                           highest_education="MSc", summary=f"{title} with {years} years.")


@pytest.fixture
def store():
    s = CandidateStore(FakeEmbedder(), client=chromadb.EphemeralClient(), prefix=f"t{uuid.uuid4().hex[:8]}")
    s.upsert("c01", _fields("Lucía Fernández Ortega", "Senior Data Scientist", "Spain", "senior", 8,
                            ["Python", "scikit-learn", "SQL"], ["Spanish", "English"]),
             "Lucía Fernández Ortega\n\nBuilt churn prediction models with scikit-learn and Airflow pipelines.")
    s.upsert("c08", _fields("Johannes Becker", "Senior Embedded Engineer", "Germany", "senior", 14,
                            ["C++", "RTOS", "AUTOSAR"], ["German", "English"]),
             "Johannes Becker\n\nFirmware for automotive ECUs, CAN bus diagnostics, ISO 26262.")
    s.upsert("c04", _fields("Anna Petrosyan", "Junior Frontend Developer", "Armenia", "junior", 1,
                            ["React", "TypeScript"], ["Armenian", "Russian", "English"]),
             "Anna Petrosyan\n\nBuilt React dashboards with TypeScript and Tailwind.")
    return s
