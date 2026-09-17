"""Country names: one canonical English name per country, so filters and extraction agree."""
from __future__ import annotations

import pycountry

# Common names the ISO list does not accept.
_ALIASES = {
    "uk": "United Kingdom", "u.k.": "United Kingdom", "england": "United Kingdom", "scotland": "United Kingdom",
    "wales": "United Kingdom", "great britain": "United Kingdom", "u.s.": "United States",
    "u.s.a.": "United States", "america": "United States", "russia": "Russian Federation",
    "korea": "South Korea", "holland": "Netherlands", "turkey": "Türkiye",
}


def canonical_country(value: str) -> str | None:
    """'USA' -> 'United States', 'germany' -> 'Germany'; None if the value is not a country."""
    v = " ".join(value.split()).strip(" .,")
    if not v:
        return None
    v = _ALIASES.get(v.casefold(), v)
    try:
        country = pycountry.countries.lookup(v)
    except LookupError:
        return None
    return getattr(country, "common_name", None) or country.name
