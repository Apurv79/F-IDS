"""Request and response contracts for ML-based attack detection."""

from typing import Literal

from pydantic import BaseModel, Field

# Labels agreed for the initial FL-IDS classification task.
AttackLabel = Literal[
    "Normal",
    "SQL Injection",
    "XSS",
    "Brute Force",
    "Other/Anomaly",
]


class PredictionRequest(BaseModel):
    """Features extracted from one HTTP request or event."""

    # Key-value features extracted by the feature pipeline.
    # Values can be numeric, Boolean, or textual at this stage.
    features: dict[str, str | int | float | bool] = Field(
        min_length=1,
        description="Extracted HTTP request features",
    )
    # Optional identifier from the caller for event correlation.
    event_id: str | None = None
    # Version of the model the caller expects to use, if specified.
    requested_model_version: str | None = None


class PredictionResponse(BaseModel):
    """Classification result returned by the ML service."""

    prediction: AttackLabel
    confidence: float = Field(ge=0.0, le=1.0)
    # Version of the model that actually produced this result.
    model_version: str
    # Optional scores for individual classes.
    class_probabilities: dict[str, float] | None = None
    # Correlation ID for tracing this detection through the system.
    request_id: str | None = None
