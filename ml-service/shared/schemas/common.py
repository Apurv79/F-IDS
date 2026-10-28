"""Common response structures reused across API endpoints."""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard format for an API error response."""

    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Safe, user-facing error message")
    request_id: str | None = Field(
        default=None,
        description="Request identifier for troubleshooting",
    )
