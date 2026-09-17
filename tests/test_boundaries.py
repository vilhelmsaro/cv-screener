"""The read path must work on any CV, so it may not reach for data only we happen to have.

seeds.py and data/profiles exist because we generate the dataset. Indexing, search and the agent must
read the PDFs alone; the seeds are the generator's input and a test answer key, never an index input.
"""
import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "cv_screener"
READ_PATH = ["index.py", "store.py", "fields.py", "agent.py", "countries.py", "cli.py"]
FORBIDDEN = {"seeds", "cv_screener.seeds"}


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level + (node.module or "")).lstrip("."))
    return names


@pytest.mark.parametrize("module", READ_PATH)
def test_read_path_does_not_import_seeds(module):
    assert not imported_modules(PACKAGE / module) & FORBIDDEN


@pytest.mark.parametrize("module", READ_PATH)
def test_read_path_does_not_read_generated_profiles(module):
    source = (PACKAGE / module).read_text(encoding="utf-8")
    assert "PROFILES_DIR" not in source and "data/profiles" not in source


def test_generator_is_the_only_place_that_uses_seeds():
    users = {p.name for p in PACKAGE.glob("*.py") if imported_modules(p) & FORBIDDEN}
    assert users == {"generate.py"}
