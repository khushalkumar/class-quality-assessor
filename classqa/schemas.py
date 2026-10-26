"""Structured-output schemas (passed to Gemini as response_schema)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Rating = Literal["Poor", "Average", "Good", "Excellent"]
DimKey = Literal["rapport_opening", "individual_engagement", "teaching_clarity",
                 "energy_warmth", "structure_closure", "professional_setup"]


# ---------------------------------------------------------------- transcription
class TSeg(BaseModel):
    start: str = Field(description="Start time relative to the start of THIS audio clip, MM:SS")
    speaker: str = Field(description="'Teacher', 'Student: <first name>' or 'Student'")
    text: str


class TranscriptChunk(BaseModel):
    language: str = Field(description="Main spoken language(s) in this clip")
    segments: list[TSeg]


# ---------------------------------------------------------------- assessment
class Evidence(BaseModel):
    timestamp: str = Field(description="MM:SS (or H:MM:SS) in the class recording")
    observation: str = Field(description="What specifically happens, in one sentence")
    quote: str = Field(description="Short verbatim quote from the transcript, or '(visual)' / '(audio)'")


class DimensionAssessment(BaseModel):
    key: DimKey
    score: int = Field(ge=1, le=5, description="1-5 using the rubric anchors")
    summary: str = Field(description="1-2 sentence justification specific to THIS class")
    evidence: list[Evidence] = Field(description="2-4 timestamped pieces of evidence")


class VisualCheck(BaseModel):
    teacher_camera_on: Literal["throughout", "mostly", "sometimes", "rarely", "unclear"]
    teacher_face_visible: Literal["clearly", "partly", "not visible", "unclear"]
    branded_background: Literal["8x8 branded", "other virtual/plain", "real room", "unclear"]
    lighting: Literal["good", "acceptable", "poor", "unclear"]
    screen_share_used: bool
    notes: str = Field(description="What the frames show, citing frame timestamps")


class Point(BaseModel):
    title: str
    detail: str
    timestamp: str


class Improvement(BaseModel):
    title: str
    detail: str = Field(description="What happened and why it matters, citing evidence")
    timestamp: str
    coaching_tip: str = Field(description="Concrete thing the teacher can do next class")


class RedFlag(BaseModel):
    issue: str = Field(description="What happened, with evidence")
    timestamp: str
    severity: Literal["minor", "major"] = Field(
        description="major ONLY for: unprofessional language/conduct towards children; teacher's own tile visibly "
                    "off/black or face not visible for most of the teaching time; or repeated A/V problems that children "
                    "complain about and that make the class hard to follow. Silence, missing homework, a late start or "
                    "a thumbnail hidden by screen share are minor at most.")


class Assessment(BaseModel):
    class_summary: str = Field(description="Topic, level, format, number of children - 2 sentences")
    visual: VisualCheck
    reviewer_concerns: list[str] = Field(
        description="BEFORE scoring: everything an 8x8 reviewer would criticise in this class, each with a timestamp")
    dimensions: list[DimensionAssessment]
    strengths: list[Point] = Field(description="3-5 specific strengths")
    improvements: list[Improvement] = Field(description="3-5 specific, actionable improvements")
    red_flags: list[RedFlag] = Field(description="Conduct, A/V or professionalism issues; empty if none")
    holistic_rating: Rating
    confidence: Literal["Low", "Medium", "High"]
    verdict: str = Field(description="3-4 sentences explaining the overall rating, referencing the decisive evidence")
