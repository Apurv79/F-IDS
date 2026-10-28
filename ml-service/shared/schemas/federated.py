"""Contracts for federated-learning rounds and client updates."""

from typing import Literal

from pydantic import BaseModel, Field


class FederatedUpdate(BaseModel):
    """Metadata and parameters submitted by a federated client."""

    # Unique identifier assigned to a participating client.
    client_id: str
    # FL round to which this update belongs.
    round_id: str
    # Version on which the client based its local training.
    # The coordinator must validate it against the round's
    # expected base model version before accepting the update.
    base_model_version: str
    # Number of training examples used locally.
    num_examples: int = Field(gt=0)
    # Model parameters serialized as numeric arrays.
    # The aggregation layer must validate shapes, lengths,
    # numeric finiteness, and resource limits before use.
    parameters: list[list[float]]
    # Optional metric values reported by the client.
    metrics: dict[str, float] = Field(default_factory=dict)


class FederatedRoundStatus(BaseModel):
    """Current status of a federated-learning round."""

    round_id: str
    status: Literal[
        "pending",
        "running",
        "aggregating",
        "completed",
        "failed",
    ]
    base_model_version: str
    # Populated when aggregation produces a new global model.
    resulting_model_version: str | None = None
    participating_clients: int = Field(ge=0)
    completed_clients: int = Field(ge=0)
