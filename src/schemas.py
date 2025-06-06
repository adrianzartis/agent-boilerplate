"""Common pydantic models used across the example agents."""

from typing import Literal

from pydantic import BaseModel


class LLMBaseModel(BaseModel):
    """Base schema for structured responses returned by language model agents."""

    model_config = {
        "description": (
            "Fields defined in subclasses are expected to be filled by the LLM."
        )
    }


class AgentRequest(BaseModel):
    """Generic request body for agents."""

    text: str


class AgentResponse(BaseModel):
    """Generic response schema from agents."""

    result: str


class SummaryOutput(LLMBaseModel):
    """Structured output for the summarising agent."""

    summary: str


class FollowupOutput(LLMBaseModel):
    """Question about the summary along with its type."""

    question: str
    kind: Literal["open-ended", "yes/no"]
