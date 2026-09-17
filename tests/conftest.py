import uuid

import chromadb
import pytest
from pydantic_ai import models

from cv_screener.store import CandidateStore

# A test that forgets to pass a fake model must fail, not spend credits. TestModel/FunctionModel still work.
models.ALLOW_MODEL_REQUESTS = False

from .fakes import FakeEmbedder, make_fields


@pytest.fixture
def store():
    # score_floor=0: the fake embedder's similarity scale is not the real model's.
    s = CandidateStore(FakeEmbedder(), client=chromadb.EphemeralClient(), prefix=f"t{uuid.uuid4().hex[:8]}",
                       score_floor=0)
    s.upsert("c01", make_fields("Lucía Fernández Ortega", "Senior Data Scientist", "Spain", "senior", 8,
                            ["Python", "scikit-learn", "SQL"], ["Spanish", "English"]),
             "Lucía Fernández Ortega\n\nBuilt churn prediction models with scikit-learn and Airflow pipelines.")
    s.upsert("c08", make_fields("Johannes Becker", "Senior Embedded Engineer", "Germany", "senior", 14,
                            ["C++", "RTOS", "AUTOSAR"], ["German", "English"]),
             "Johannes Becker\n\nFirmware for automotive ECUs, CAN bus diagnostics, ISO 26262.")
    s.upsert("c04", make_fields("Anna Petrosyan", "Junior Frontend Developer", "Armenia", "junior", 1,
                            ["React", "TypeScript"], ["Armenian", "Russian", "English"]),
             "Anna Petrosyan\n\nBuilt React dashboards with TypeScript and Tailwind.")
    return s
