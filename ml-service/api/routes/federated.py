"""Federated learning coordinator routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies.auth import verify_internal_api_key
from federated.rounds.coordinator import FederatedCoordinator
from ml.dataset.synthetic import generate_http_samples
from shared.schemas.federated import FederatedRoundStatus, FederatedUpdate

router = APIRouter(
    prefix="/api/v1/federated",
    tags=["Federated Learning"],
    dependencies=[Depends(verify_internal_api_key)],
)

# Shared coordinator instance
_coordinator = FederatedCoordinator(initial_model_version="v1.0.0")


class SimulationTriggerRequest(BaseModel):
    num_clients: int = Field(default=3, ge=2, le=20)
    partition_strategy: str = Field(default="iid", pattern="^(iid|non_iid)$")
    sample_size_per_class: int = Field(default=30, ge=10, le=500)


@router.post(
    "/rounds/start",
    response_model=FederatedRoundStatus,
    summary="Start a new Federated Learning Round",
)
async def start_federated_round() -> FederatedRoundStatus:
    """Initialize a new round pinned to the active global model version."""
    try:
        active_round = _coordinator.start_round()
        return FederatedRoundStatus(
            round_id=active_round.round_id,
            status="running",
            base_model_version=active_round.base_model_version,
            participating_clients=0,
            completed_clients=0,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "ROUND_CONFLICT", "message": str(exc)},
        ) from exc


@router.post(
    "/rounds/update",
    summary="Submit Client Parameter Weights",
)
async def submit_client_update(update: FederatedUpdate) -> dict:
    """Submit local client model parameters for round aggregation."""
    try:
        _coordinator.submit_update(
            client_id=update.client_id,
            base_model_version=update.base_model_version,
            parameters=update.parameters,
            num_examples=update.num_examples,
            metrics=update.metrics,
        )
        return {
            "status": "accepted",
            "client_id": update.client_id,
            "round_id": update.round_id,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_UPDATE", "message": str(exc)},
        ) from exc


@router.post(
    "/rounds/aggregate",
    summary="Run FedAvg Aggregation on Round Updates",
)
async def aggregate_round() -> dict:
    """Aggregate all accepted updates and publish new global model version."""
    try:
        return _coordinator.aggregate()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AGGREGATION_FAILED", "message": str(exc)},
        ) from exc


@router.post(
    "/simulate",
    summary="Trigger End-to-End Multi-Client Federated Simulation",
)
async def simulate_federated_round(request: SimulationTriggerRequest) -> dict:
    """Run an automated multi-client simulation with IID or Non-IID partitioning."""
    dataset = generate_http_samples(
        rows_per_class=request.sample_size_per_class,
        seed=42,
    )
    result = _coordinator.run_simulation(
        dataset=dataset,
        num_clients=request.num_clients,
        partition_strategy=request.partition_strategy,
    )
    return result
