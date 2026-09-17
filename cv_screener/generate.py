"""Step 1: generate synthetic candidates -> JSON profile, photo, PDF CV."""
from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic_ai import ModelRetry
from rich.console import Console

from .config import CVS_DIR, PHOTOS_DIR, PROFILES_DIR, settings
from .fields import parse_languages, seniority_for, years_from_text
from .llm import generate_image, structured_agent
from .models import CandidateProfile
from .seeds import SEEDS

console = Console()
TEMPLATES = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=select_autoescape(["html"]),
)

GEN_INSTRUCTIONS = """You write realistic CVs for synthetic candidates.
Stay strictly consistent with the given facts (name, role, level, years, location, languages, stack, education).
Make it read like a real person's CV, not a template:
- real or plausible companies for the region, believable job progression and dates that add up to the stated years
- specific bullets with concrete numbers, some short, some long; not every bullet has a metric
- small human quirks: an older unrelated job, a short gap or a side project, uneven section lengths
- juniors get 1-2 roles and a short CV, seniors 3-5 roles
- the current (first) job title matches the level: no "Senior", "Lead" or "Staff" for a mid-level person
- the summary states the years exactly, e.g. "with 6 years of experience"
- email and phone in the local format; links look plausible (linkedin/github/portfolio as fits the role)
- do NOT add spoken languages or core technologies beyond those given (minor tools are fine)."""


def seed_mismatches(seed: dict, profile: CandidateProfile) -> list[str]:
    """Seed facts the CV gets wrong, judged by the same rules indexing uses to read them back from the PDF."""
    problems = []
    title = profile.experience[0].title if profile.experience else ""
    if (level := seniority_for(title, seed["years"])) != seed["level"]:
        problems.append(f"the current job title {title!r} reads as {level}, but the person is {seed['level']}")
    if (years := years_from_text(profile.summary)) != seed["years"]:
        problems.append(f"the summary must say \"{seed['years']} years of experience\" (found: {years})")
    expected = {lang.casefold() for lang in parse_languages([seed["languages"]])}
    if (actual := {lang.name.casefold() for lang in profile.languages}) != expected:
        problems.append(f"spoken languages must be exactly {sorted(expected)} (found: {sorted(actual)})")
    return problems


def _profile(seed: dict) -> CandidateProfile:
    agent = structured_agent(settings.gen_model, CandidateProfile, GEN_INSTRUCTIONS, max_tokens=8000)

    @agent.output_validator
    def matches_seed(profile: CandidateProfile) -> CandidateProfile:
        # The model gets the problems back and rewrites the CV; after the retries run out the candidate fails.
        if problems := seed_mismatches(seed, profile):
            raise ModelRetry("Fix these facts and return the whole CV again: " + "; ".join(problems))
        return profile

    facts = "\n".join(f"{k}: {v}" for k, v in seed.items() if k not in {"id", "template", "photo"})
    # Without today's date the model cannot make "Present" roles and total years add up.
    today = date.today().strftime("%B %Y")
    return agent.run_sync(f"Today is {today}. Write the CV for this person.\n{facts}").output


def _photo(seed: dict, path: Path) -> None:
    prompt = (
        f"Headshot photo for a CV: {seed['photo']}, working as {seed['role']}. "
        "Head and shoulders, realistic photograph, not illustrated, no text, square framing."
    )
    path.write_bytes(generate_image(prompt))


def _render(seed: dict, profile: CandidateProfile, photo: Path, out: Path) -> None:
    from weasyprint import HTML  # imported lazily: needs system libs

    data = photo.read_bytes()
    mime = "image/jpeg" if data.startswith(b"\xff\xd8\xff") else "image/png"  # image models return either
    photo_uri = f"data:{mime};base64," + base64.b64encode(data).decode()
    html = TEMPLATES.get_template(f"{seed['template']}.html").render(p=profile, photo=photo_uri)
    HTML(string=html).write_pdf(out)


def _build(seed: dict, force: bool) -> Path:
    pid = seed["id"]
    prof_path, photo_path, pdf_path = PROFILES_DIR / f"{pid}.json", PHOTOS_DIR / f"{pid}.png", CVS_DIR / f"{pid}.pdf"
    if force or not prof_path.exists():  # cache: reruns cost nothing
        prof_path.write_text(_profile(seed).model_dump_json(indent=2), encoding="utf-8")
    profile = CandidateProfile.model_validate(json.loads(prof_path.read_text(encoding="utf-8")))

    if force or not photo_path.exists():
        _photo(seed, photo_path)

    _render(seed, profile, photo_path, pdf_path)
    return pdf_path


def run(force: bool = False, only: list[str] | None = None) -> None:
    for d in (PROFILES_DIR, PHOTOS_DIR, CVS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    failed = []
    for seed in SEEDS:
        if only and seed["id"] not in only:
            continue
        console.print(f"[bold]{seed['id']}[/] {seed['name']} - {seed['role']}")
        try:  # one failed candidate should not lose the rest; cached steps make a rerun cheap
            pdf_path = _build(seed, force)
        except Exception as e:
            failed.append(seed["id"])
            console.print(f"  [red]failed:[/] {e!r}")
            continue
        console.print(f"  -> {pdf_path.relative_to(CVS_DIR.parent.parent)}")
    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}. Rerun `cvs generate --only <id>` for each.")
