"""Structured fields read from CV text with plain rules: same PDF in, same fields out.

Works on the chunks from store.chunk_cv, so each rule only looks at the section where a fact is printed.
Only the country can stay unresolved (returned in `unresolved`); index.py asks the LLM for that one field.
"""
from __future__ import annotations

import re
from datetime import date

from .countries import canonical_country
from .parsing import split_top_level
from .models import ExtractedFields
from .store import HEADER, MONTH, Chunk, date_line_index, fold

# First match wins, checked against the current job title.
_SENIORITY_WORDS = [
    ("intern", r"intern|internship|trainee|working student"),
    ("junior", r"junior|jr|graduate|entry[- ]level"),
    ("manager", r"manager|director|head of|vp"),
    ("staff", r"staff|principal|distinguished"),
    ("lead", r"lead"),
    ("senior", r"senior|sr"),
]
_DEGREE_RANKS = [
    (4, r"ph\.?d|doctor|doctorate|dphil"),
    (3, r"master|m\.?sc|m\.?s|mba|m\.?eng|m\.?a|dipl"),
    (2, r"bachelor|b\.?sc|b\.?s|b\.?a|b\.?eng|b\.?tech|ingenier|licen"),
]
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
_YEARS = re.compile(rf"\b(\d{{1,2}}|{'|'.join(_NUMBER_WORDS)})\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_POINT = rf"(?:({MONTH})\s+|(\d{{1,2}})[./])?(\d{{4}})"
_RANGE = re.compile(rf"{_POINT}\s*[–—-]\s*(?:{_POINT}|(present|current|now|today))", re.IGNORECASE)
_LANGUAGE_LEVEL = re.compile(r"^(?:[abc][12]|native|fluent|basic|intermediate|advanced|professional|conversational)",
                             re.IGNORECASE)
_SKILL_LABEL = re.compile(r"^([^,:()]{1,40}):\s*(.*)$")


def body(chunk: Chunk) -> list[str]:
    """Chunk lines without the printed section heading."""
    lines = chunk.text.splitlines()
    return lines if chunk.section == HEADER else lines[1:]


def join_wrapped(text: str, line: str) -> str:
    """Undo a PDF line wrap: "Cross-" + "functional" -> "Cross-functional", otherwise join with a space."""
    if not text:
        return line.strip()
    if re.search(r"\w[-/]$", text):
        return text + line.strip()
    return f"{text} {line.strip()}"


def parse_skills(lines: list[str]) -> list[str]:
    """Items of a Skills section.

    A line starting with "Label:" opens a comma-separated group, and following lines continue it (narrow
    layouts wrap mid-item: "Selenium" / "WebDriver"). Without labels, each line is its own list unless it
    clearly continues the previous one (starts lowercase or with "(", or the previous ends with "-", "/", "&").
    """
    groups: list[str] = []
    labeled = False
    for line in (line.strip() for line in lines):
        if m := _SKILL_LABEL.match(line):
            groups.append(m.group(2).strip())
            labeled = True
        elif groups and (labeled or re.match(r"[a-z(]", line) or re.search(r"[-/&]$", groups[-1])):
            groups[-1] = join_wrapped(groups[-1], line)
        else:
            groups.append(line)
    return _dedupe(item for group in groups for item in split_top_level(group))


def parse_languages(lines: list[str]) -> list[str]:
    """Language names from "Spanish — Native" lines or "Polish (Native), English (B2)" lists."""
    names = []
    for item in split_top_level(",".join(lines), separators=",;|·•"):  # real CVs also separate with "|"
        name = re.split(r"\s+[—–-](?:\s+|$)|\s*[(:]", item)[0].strip()
        if name and name[0].isalpha() and len(name) <= 30 and not _LANGUAGE_LEVEL.match(name):
            names.append(name)
    return _dedupe(names)


def find_location(chunks: list[Chunk]) -> str | None:
    """The "City, Country" text from the contact line (header) or a Contact section."""
    lines = []
    for c in chunks:
        if c.section == "contact":
            lines += body(c)
        elif c.section == HEADER:
            lines += [line for line in c.text.splitlines() if "@" in line]
    segments = [s.strip() for line in lines for s in re.split(r"\s+[·•|]\s+|\s*\|\s*", line)]
    places = [s for s in segments
              if len(s) >= 3 and s[0].isupper() and not re.search(r"[\d@./=]", s)]
    with_comma = [s for s in places if "," in s]
    if with_comma:
        return with_comma[0]
    return next((s for s in places if canonical_country(s)), None)


def split_location(location: str | None) -> tuple[str, str | None]:
    """"Bogotá, Colombia (Remote)" -> ("Bogotá", "Colombia"); the country is None if not recognised."""
    if not location:
        return "", None
    parts = [p.strip() for p in re.sub(r"\(.*?\)", "", location).split(",") if p.strip()]
    if not parts:
        return "", None
    return parts[0], canonical_country(parts[-1])


def years_from_text(text: str) -> int | None:
    """Years of experience as the CV states them: "with 8 years of experience"."""
    if m := _YEARS.search(text):
        word = m.group(1).lower()
        return _NUMBER_WORDS.get(word) or int(word)
    return None


def _month_index(month: str | None, number: str | None, year: str, default: int) -> int:
    if month:
        m = _MONTHS.index(month[:3].lower()) + 1
    elif number:
        m = int(number)
    else:
        m = default
    return int(year) * 12 + m - 1


def years_from_dates(entries: list[Chunk], today: date) -> int:
    """Total experience from the job date ranges, with overlapping jobs counted once."""
    spans = []
    for chunk in entries:
        lines = body(chunk)
        idx = date_line_index("\n".join(lines))
        # A date line of its own is the usual layout; otherwise look for dates inside the first lines.
        text = " ".join(lines[idx:idx + 2] if idx is not None else lines[:2])  # a range can wrap
        if not (m := _RANGE.search(text)):
            continue
        start = _month_index(m.group(1), m.group(2), m.group(3), default=1)
        if m.group(7):
            end = today.year * 12 + today.month - 1
        else:
            end = _month_index(m.group(4), m.group(5), m.group(6), default=1)
        if end > start:
            spans.append((start, end))
    months, reach = 0, None
    for start, end in sorted(spans):
        if reach is not None and start < reach:
            start = reach
        if end > start:
            months += end - start
        reach = max(reach or end, end)
    return round(months / 12)


def entry_title(chunk: Chunk) -> str:
    """Title of a job or degree entry: the text above its date line, or the first line without its dates.

    CVs that print "Software Engineer, Acme (Jan 2019 - Mar 2021)" on one line keep only the part before
    the dates.
    """
    lines = body(chunk)
    idx = date_line_index("\n".join(lines))
    if idx:
        return " ".join(line.strip() for line in lines[:idx])
    first = lines[0] if lines else ""
    if _RANGE.search(first):
        first = _RANGE.split(first)[0]
    # "Software Engineer | 10Web | Yerevan" -> the role; a comma can be part of the role, a pipe cannot.
    return re.split(r"\s*[|·•]\s*", first.strip(" ,;-–—([|"))[0].strip()


def seniority_for(title: str, years: int) -> str:
    t = fold(title)
    for level, pattern in _SENIORITY_WORDS:
        if re.search(rf"\b(?:{pattern})\b", t):
            return level
    return "junior" if years < 2 else "mid" if years < 7 else "senior"


def highest_degree(education: list[Chunk]) -> str:
    """Highest degree by level (PhD > Master > Bachelor > other); the first listed wins a tie."""
    titles = [entry_title(c) for c in education]

    def rank(title: str) -> int:
        return next((r for r, p in _DEGREE_RANKS if re.search(rf"\b(?:{p})\b", fold(title))), 1)

    return max(titles, key=rank, default="")


def extract_fields(chunks: list[Chunk], name: str, today: date) -> tuple[ExtractedFields, list[str]]:
    """Fields read from the page. Returns the fields and the names of fields the rules could not resolve."""
    by_section: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_section.setdefault(c.section, []).append(c)
    header = by_section.get(HEADER, [])
    header_lines = [line for c in header for line in body(c)]

    summary_chunks = by_section.get("summary") or header
    summary_lines = [line for c in summary_chunks for line in body(c)
                     if fold(line).strip() != fold(name).strip()]
    summary = ""
    for line in summary_lines:
        summary = join_wrapped(summary, line)

    experience = by_section.get("experience", [])
    if experience:
        title = entry_title(experience[0])
    else:  # no dated jobs: fall back to the headline under the name
        headline = next((line for line in header_lines[1:2]), "")
        title = re.split(r"\s+[|—–]\s+", headline)[0].strip()

    years = years_from_text(summary)
    if years is None:
        years = years_from_dates(experience, today)

    city, country = split_location(find_location(chunks))
    fields = ExtractedFields(
        full_name=name,
        current_title=title,
        city=city,
        country=country or "",
        seniority=seniority_for(title, years),
        years_experience=years,
        skills=[s for c in by_section.get("skills", []) for s in parse_skills(body(c))],
        languages=[lang for c in by_section.get("languages", []) for lang in parse_languages(body(c))],
        highest_education=highest_degree(by_section.get("education", [])),
        summary=summary[:1000],
    )
    return fields, ([] if country else ["country"])


def _dedupe(items) -> list[str]:
    seen, out = set(), []
    for item in items:
        if (key := fold(item)) not in seen:
            seen.add(key)
            out.append(item)
    return out
