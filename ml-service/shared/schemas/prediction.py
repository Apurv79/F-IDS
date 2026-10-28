"""Request and response contracts for ML-based attack detection."""

from typing import Any, Literal
from pydantic import BaseModel, Field

# Labels agreed for the initial FL-IDS classification task.
AttackLabel = Literal[
    "Normal",
    "SQL Injection",
    "XSS",
    "Brute Force",
    "Other/Anomaly",
]


class RawHttpRequestData(BaseModel):
    """Raw HTTP request data payload forwarded from Node.js Express."""

    url: str = Field(description="Target path and query string, e.g. /search?q=test")
    method: str = Field(default="GET", description="HTTP verb (GET, POST, PUT, DELETE, etc.)")
    headers: dict[str, str] = Field(default_factory=dict, description="Sanitized HTTP headers (User-Agent, etc.)")
    body: str | dict[str, Any] = Field(default="", description="Request payload body (string or parsed JSON)")
    ip_request_rate: float = Field(default=0.0, ge=0.0, description="Requests per minute from client IP (from Redis)")
    failed_login_count: float = Field(default=0.0, ge=0.0, description="Consecutive failed logins for user/IP")


class PredictionRequest(BaseModel):
    """Prediction request accepted by POST /api/v1/predict.

    Accepts either raw request details (forwarded directly from Node.js)
    or pre-extracted numeric/textual features.
    """

    event_id: str | None = Field(default=None, description="Unique idempotency ID from Node.js event bus")
    raw_request: RawHttpRequestData | None = Field(default=None, description="Raw incoming HTTP event details")
    features: dict[str, str | int | float | bool] | None = Field(
        default=None,
        description="Pre-extracted features if caller extracted them",
    )
    requested_model_version: str | None = Field(default=None, description="Optional target model version")


class PredictionResponse(BaseModel):
    """Classification result returned by the ML service."""

    prediction: AttackLabel
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    class_probabilities: dict[str, float] | None = None
    event_id: str | None = None
    request_id: str | None = None
