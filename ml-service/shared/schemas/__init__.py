"""Shared Pydantic schemas for FL-IDS request and response contracts."""

from shared.schemas.common import ErrorResponse
from shared.schemas.explanation import ExplanationRequest, ExplanationResponse
from shared.schemas.federated import (
    FederatedRoundStatus,
    FederatedUpdate,
)
from shared.schemas.prediction import PredictionRequest, PredictionResponse
from shared.schemas.summarization import (
    SummarizationRequest,
    SummarizationResponse,
)
from shared.schemas.training import TrainingRequest, TrainingResponse

__all__ = [
    "ErrorResponse",
    "PredictionRequest",
    "PredictionResponse",
    "ExplanationRequest",
    "ExplanationResponse",
    "SummarizationRequest",
    "SummarizationResponse",
    "TrainingRequest",
    "TrainingResponse",
    "FederatedUpdate",
    "FederatedRoundStatus",
]
