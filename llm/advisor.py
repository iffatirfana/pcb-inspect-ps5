"""Shared advisory types: the validated Advice model and the pluggable Backend.

Llama never decides PASS/REVIEW/REWORK/REJECT - the rule engine does.
The model call is pluggable through Backend.generate.
"""
from typing import List, Literal

from pydantic import BaseModel, field_validator


class Advice(BaseModel):
    summary: str
    risk: str
    actions: List[str]
    verdict: Literal["PASS", "REVIEW", "REWORK", "REJECT", "INVALID_CAPTURE"]

    @field_validator("actions")
    @classmethod
    def short_actions(cls, v):
        return [a.strip() for a in v if a and a.strip()][:8]


class Backend:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
