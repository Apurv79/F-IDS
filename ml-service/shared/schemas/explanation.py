"""Contracts for AI-generated attack explanations."""

from pydantic import BaseModel, Field


class ExplanationRequest(BaseModel):
    """Information needed to explain a detection."""

    # Predicted class produced by the ML detection model.
    prediction: str
    # Features that contributed to or describe the detection.
    features: dict[str, str | int | float | bool]
    # Optional confidence score from the ML model.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # Optional model version for traceability.
    model_version: str | None = None


class ExplanationResponse(BaseModel):
    """Explanation and remediation guidance for a detection."""

    explanation: str
    # Practical defensive steps, not executable attack instructions.
    recommendations: list[str] = Field(default_factory=list)
    # Provider used to generate the explanation, e.g. huggingface or ollama.
    provider: str
    # True when a fallback or degraded response was used.
    degraded: bool = False
