"""Contracts for centralized model-training jobs."""

from typing import Literal

from pydantic import BaseModel, Field

# Algorithms approved for the initial implementation.
ModelAlgorithm = Literal[
    "logistic_regression",
    "random_forest",
    "mlp",
]


class TrainingRequest(BaseModel):
    """Configuration for a training job."""

    # Dataset identifier or registered dataset path.
    # Dataset resolution and path safety will be handled separately.
    dataset_id: str
    # Model family to train.
    algorithm: ModelAlgorithm = "logistic_regression"
    # Data partition strategy for evaluation.
    split_strategy: Literal["iid", "non_iid"] = "iid"
    # Reproducible random seed.
    random_seed: int = 42
    # Whether to calculate additional evaluation metrics.
    evaluate: bool = True


class TrainingResponse(BaseModel):
    """Summary returned after a training job completes."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    # Filled in after training finishes successfully.
    model_version: str | None = None
    # Metrics such as accuracy, precision, recall and F1.
    metrics: dict[str, float] | None = None
    # Safe summary of a failure, if applicable.
    error: str | None = None
