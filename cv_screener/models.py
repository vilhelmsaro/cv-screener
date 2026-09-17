"""Pydantic schemas: what the generator writes and what the indexer extracts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Seniority = Literal["intern", "junior", "mid", "senior", "lead", "staff", "manager"]


class Seed(BaseModel):
    """Fixed facts for one synthetic candidate (seeds.py). The generator must not contradict them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^c\d{2}$")
    name: str
    role: str
    level: Seniority
    years: int = Field(ge=0, le=50)
    location: str
    languages: str = Field(description="e.g. 'Spanish (native), English (C1)'")
    stack: str
    education: str
    style: str = Field(description="how the CV should read, e.g. 'German Lebenslauf style, formal'")
    template: Literal["classic", "modern", "compact"]
    photo: str = Field(description="who is in the headshot and the setting")


class Experience(BaseModel):
    title: str
    company: str
    location: str
    start: str = Field(description="e.g. 'Mar 2019' or '03/2019'")
    end: str = Field(description="e.g. 'Present' or 'Jun 2022'")
    bullets: list[str]


class Education(BaseModel):
    degree: str
    institution: str
    location: str
    start: str
    end: str
    details: str | None = None


class Language(BaseModel):
    name: str
    level: str = Field(description="e.g. 'Native', 'C1', 'Professional working proficiency'")


class SkillGroup(BaseModel):
    group: str
    items: list[str]


class Project(BaseModel):
    name: str
    description: str


class CandidateProfile(BaseModel):
    """Full CV content produced by the generator."""

    full_name: str
    headline: str
    email: str
    phone: str
    location: str
    links: list[str]
    summary: str
    experience: list[Experience]
    education: list[Education]
    skills: list[SkillGroup]
    languages: list[Language]
    certifications: list[str] = []
    projects: list[Project] = []
    interests: list[str] = []


class ExtractedFields(BaseModel):
    """Structured fields the indexer extracts back out of the PDF text."""

    full_name: str
    current_title: str
    city: str
    country: str
    seniority: Seniority
    years_experience: int = Field(description="Total years of professional experience, rounded")
    skills: list[str] = Field(description="Technologies, tools and methods, one per item")
    languages: list[str] = Field(description="Spoken languages only, names in English, e.g. 'Spanish'")
    highest_education: str
    summary: str = Field(description="2-3 sentence neutral profile summary")
