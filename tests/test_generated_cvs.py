"""Checks chunking on the real generated PDFs (data/cvs), using the generator's JSON as the answer key.

The JSON is never used for indexing; here it only tells us what each PDF is known to contain. Skipped on a
fresh clone, before `cvs generate` has run. Offline and deterministic: no API key, no LLM.
"""
import json
import re
from pathlib import Path

import pytest

from cv_screener.config import CVS_DIR, PROFILES_DIR
from cv_screener.index import pdf_text
from cv_screener.models import CandidateProfile
from cv_screener.store import chunk_cv, fold, missing_sections

PAIRS = [(pdf, PROFILES_DIR / f"{pdf.stem}.json") for pdf in sorted(CVS_DIR.glob("*.pdf"))]
PAIRS = [(pdf, js) for pdf, js in PAIRS if js.exists()]

pytestmark = pytest.mark.skipif(not PAIRS, reason="no generated CVs; run `cvs generate` first")


def flat(text: str) -> str:
    """Undo PDF line wrapping so text can be compared with the source JSON."""
    text = re.sub(r"(\w)([-/])\n(\w)", r"\1\2\3", text)  # "click-\nthrough", "A/\nB" -> joined
    return " ".join(fold(text).split())


@pytest.mark.parametrize("pdf, profile_json", PAIRS, ids=[p.stem for p, _ in PAIRS])
def test_every_section_and_entry_is_chunked(pdf: Path, profile_json: Path):
    profile = CandidateProfile.model_validate(json.loads(profile_json.read_text(encoding="utf-8")))
    chunks = chunk_cv(pdf_text(pdf), name=profile.full_name)
    by_section: dict[str, list[str]] = {}
    for c in chunks:
        by_section.setdefault(c.section, []).append(flat(c.text))

    assert missing_sections({s: len(v) for s, v in by_section.items()}) == []
    assert len(by_section["experience"]) == len(profile.experience)
    assert len(by_section["education"]) == len(profile.education)
    for key, present in [("projects", profile.projects), ("certifications", profile.certifications),
                         ("interests", profile.interests)]:
        assert (key in by_section) == bool(present), key

    # Each job is one chunk holding its own title, company and every bullet.
    for job in profile.experience:
        matches = [c for c in by_section["experience"] if flat(f"{job.title} {job.start}") in c and flat(job.company) in c]
        assert len(matches) == 1, f"{job.title} @ {job.company}"
        for bullet in job.bullets:
            assert flat(bullet) in matches[0], bullet[:60]

    assert flat(profile.summary) in flat("\n".join(c.text for c in chunks))
