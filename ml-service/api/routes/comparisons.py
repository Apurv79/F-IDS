from fastapi import APIRouter, Depends, HTTPException, Request

from api.contracts import (
    ModelComparisonRequest,
    ModelComparisonResponse,
)
from api.dependencies import require_internal_api_key, service

router = APIRouter(
    prefix="/comparisons",
    tags=["Evaluation comparisons"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.post(
    "/centralized-vs-federated",
    response_model=ModelComparisonResponse,
    summary="Compare centralized and federated results",
)
async def compare_models(
    payload: ModelComparisonRequest,
    request: Request,
):
    svc = service(request, "experiments")
    if not hasattr(svc, "compare"):
        raise HTTPException(
            status_code=501,
            detail={
                "code": "OPERATION_NOT_IMPLEMENTED",
                "message": "Experiment service must implement compare().",
            },
        )
    result = await svc.compare(payload)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "EXPERIMENT_NOT_FOUND",
                "message": (
                    "One or both experiments were not found."
                ),
            },
        )
    return result
