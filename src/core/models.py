from typing import Literal
from pydantic import BaseModel, Field


class Classification(BaseModel):
    """Classification model for helpdesk queries."""

    category: Literal["automatic", "escalated"] = Field(
        description="Whether the query can be resolved automatically or needs escalation"
    )
    justification: str = Field(
        description="Brief justification for the classification (max 20 words)"
    )
