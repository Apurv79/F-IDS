"""Contracts for AI-assisted security log summarization."""

from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    """Input log records to summarize."""

    # Security events or log lines selected for summarization.
    logs: list[str] = Field(min_length=1, max_length=500)
    # Optional instruction describing the summary focus.
    # For example: summarize repeated failed login attempts.
    focus: str | None = None


class SummarizationResponse(BaseModel):
    """Structured summary of security events."""

    summary: str
    # Key observations, such as repeated login failures.
    key_findings: list[str] = Field(default_factory=list)
    # Count of records actually included in the summary.
    records_processed: int = Field(ge=0)
    # AI provider that generated the result.
    provider: str
    # Indicates fallback or reduced functionality.
    degraded: bool = False
