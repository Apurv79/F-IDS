from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from api.contracts import (
    AggregateRequest,
    FederatedRoundCreate,
    FederatedRoundResponse,
    FederatedUpdateAccepted,
    FederatedUpdateRequest,
)
from api.dependencies import require_internal_api_key, service

router = APIRouter(
    prefix="/federated",
    tags=["Federated Learning"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.post(
    "/rounds",
    response_model=FederatedRoundResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new federated round",
)
async def create_round(payload: FederatedRoundCreate, request: Request):
    svc = service(request, "federated")
    return await svc.create_round(payload)


@router.get(
    "/rounds/{round_id}",
    response_model=FederatedRoundResponse,
    summary="Get status of a federated round",
)
async def get_round(round_id: str, request: Request):
    svc = service(request, "federated")
    result = await svc.get_round(round_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ROUND_NOT_FOUND",
                "message": "Federated round was not found.",
            },
        )
    return result


@router.post(
    "/rounds/{round_id}/updates",
    response_model=FederatedUpdateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a client model update",
)
async def submit_update(
    round_id: str,
    payload: FederatedUpdateRequest,
    request: Request,
):
    if payload.round_id != round_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "ROUND_ID_MISMATCH",
                "message": "Path round_id must match body round_id.",
            },
        )
    svc = service(request, "federated")
    return await svc.submit_update(payload)


@router.post(
    "/rounds/{round_id}/aggregate",
    response_model=FederatedRoundResponse,
    summary="Trigger FL aggregation",
)
async def aggregate_round(
    round_id: str,
    payload: AggregateRequest,
    request: Request,
):
    svc = service(request, "federated")
    result = await svc.aggregate(round_id, force=payload.force)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ROUND_NOT_FOUND",
                "message": "Federated round was not found.",
            },
        )
    return result
