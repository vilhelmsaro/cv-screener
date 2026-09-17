import uuid

import chromadb
import pytest

from cv_screener.store import CandidateStore

from .fakes import FakeEmbedder, make_fields


@pytest.fixture
def store():
    s = CandidateStore(FakeEmbedder(), client=chromadb.EphemeralClient(), prefix=f"t{uuid.uuid4().hex[:8]}")
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
