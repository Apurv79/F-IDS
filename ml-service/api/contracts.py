"""Public Backend <-> ML service contract.

Shared request and response models used by FastAPI and Node.js.
"""

from datetime import datetime, timezone
import math
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.dataset.labels import CANONICAL_LABELS

# Canonical attack classes
AttackLabel = Literal[
    "Normal",
    "SQL Injection",
    "XSS",
    "Brute Force",
    "Other/Anomaly",
]


class StrictModel(BaseModel):
    """Base model enforcing strict schemas and forbid extra attributes."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --- 1. Health & Status ---
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "not_ready"]
    service: str = "fl-ids-ml-service"
    version: str
    checks: dict[
        str,
        Literal["ok", "degraded", "unavailable", "not_configured"],
    ] = Field(default_factory=dict)


# --- 2. Ingress & Prediction ---
class RequestContext(StrictModel):
    """Sanitized HTTP request data used for detection.

    Do not send cookies, authorization tokens, or unnecessary PII.
    """

    method: str = Field(default="GET", min_length=1, max_length=16)
    url: str = Field(min_length=1, max_length=8192)
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = Field(default=None, max_length=100_000)
    user_agent: str | None = Field(default=None, max_length=2048)
    source_ip: str | None = Field(default=None, max_length=64)
    request_rate: float | None = Field(default=None, ge=0)
    failed_login_count: int | None = Field(default=None, ge=0)


class PredictionRequest(StrictModel):
    event_id: str | None = Field(default=None, max_length=128)
    node_id: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional requested model version. Null means the active "
            "approved model."
        ),
    )
    request: RequestContext | None = None
    features: dict[str, str | int | float | bool] | None = None

    @model_validator(mode="after")
    def require_input(self):
        if (self.request is None) == (self.features is None):
            raise ValueError(
                "Provide exactly one of 'request' or 'features'. "
                "Prefer 'request' so the ML service owns feature extraction."
            )
        return self


class PredictionResponse(BaseModel):
    event_id: str | None = None
    prediction: AttackLabel
    confidence: float = Field(ge=0, le=1)
    model_version: str
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_contributions: list[dict[str, Any]] = Field(default_factory=list)


# --- 3. AI Explanation & Summarization ---
class ExplainRequest(StrictModel):
    event_id: str | None = None
    prediction: AttackLabel
    confidence: float = Field(ge=0, le=1)
    model_version: str
    features: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ExplainResponse(BaseModel):
    explanation: str
    recommendations: list[str] = Field(default_factory=list)
    provider: str
    degraded: bool = False


class SummarizeRequest(StrictModel):
    logs: list[str] = Field(min_length=1, max_length=500)
    focus: str | None = None
    time_window_start: str | None = None
    time_window_end: str | None = None

    @model_validator(mode="after")
    def validate_window(self):
        if self.time_window_start and self.time_window_end:
            if self.time_window_start > self.time_window_end:
                raise ValueError("time_window_start cannot be after time_window_end")
        return self


class SummarizeResponse(BaseModel):
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    records_processed: int = Field(ge=0)
    provider: str
    degraded: bool = False


# --- 4. Centralized Training ---
class TrainingRequest(StrictModel):
    dataset_id: str
    algorithm: Literal["logistic_regression", "random_forest", "mlp"] = "logistic_regression"
    split_strategy: Literal["iid", "non_iid"] = "iid"
    random_seed: int = 42
    evaluate: bool = True


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    model_version: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None


# --- 5. Models & Artifacts ---
class ModelMetadata(BaseModel):
    model_version: str
    algorithm: str
    dataset_version: str
    split_strategy: str
    feature_schema_version: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str
    random_seed: int
    created_at: str


class ModelDownloadResponse(BaseModel):
    model_version: str
    download_url: str
    expires_in_seconds: int = 3600


# --- 6. Federated Learning ---
class FederatedRoundCreate(StrictModel):
    num_clients: int = Field(default=3, ge=2, le=50)
    min_clients: int = Field(default=2, ge=2, le=50)
    partition_strategy: Literal["iid", "non_iid"] = "iid"
    sample_size_per_class: int = Field(default=30, ge=10, le=500)

    @model_validator(mode="after")
    def validate_client_bounds(self):
        if self.min_clients > self.num_clients:
            raise ValueError("min_clients cannot exceed num_clients")
        return self


class FederatedRoundResponse(BaseModel):
    round_id: str
    status: Literal["pending", "running", "aggregating", "completed", "failed"]
    base_model_version: str
    resulting_model_version: str | None = None
    participating_clients: int = Field(default=0, ge=0)
    completed_clients: int = Field(default=0, ge=0)
    metrics: dict[str, float] | None = None


class FederatedUpdateRequest(StrictModel):
    client_id: str = Field(min_length=1, max_length=128)
    round_id: str = Field(min_length=1, max_length=128)
    base_model_version: str = Field(min_length=1, max_length=128)
    num_examples: int = Field(gt=0)
    parameters: list[list[float]]
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_params_finite(self):
        for sub_array in self.parameters:
            for val in sub_array:
                if math.isnan(val) or math.isinf(val):
                    raise ValueError("Federated update parameters must contain finite numeric values.")
        return self


class FederatedUpdateAccepted(BaseModel):
    status: Literal["accepted", "rejected"] = "accepted"
    client_id: str
    round_id: str


class AggregateRequest(StrictModel):
    force: bool = False


# --- 7. Datasets ---
class DatasetMetadata(BaseModel):
    dataset_id: str
    dataset_version: str
    file_name: str
    sha256: str
    rows: int
    columns: int
    label_column: str
    created_at: str


class DatasetValidationRequest(StrictModel):
    dataset_id: str
    label_column: str = "label"
    required_columns: list[str] | None = None


class DatasetValidationResponse(BaseModel):
    is_valid: bool
    summary: dict[str, Any]
    error: str | None = None


# --- 8. Experiments & Comparisons ---
class ExperimentSummary(BaseModel):
    experiment_id: str
    kind: str
    algorithm: str
    dataset_version: str
    metrics: dict[str, float]
    created_at: str


class ModelComparisonRequest(StrictModel):
    centralized_experiment_id: str
    federated_experiment_id: str


class ModelComparisonResponse(BaseModel):
    centralized_metrics: dict[str, float]
    federated_metrics: dict[str, float]
    accuracy_delta: float
    f1_delta: float
    winner: str
