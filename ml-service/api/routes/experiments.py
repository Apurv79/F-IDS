from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.contracts import ExperimentSummary
from api.dependencies import require_internal_api_key, service

router = APIRouter(
    prefix="/experiments",
    tags=["Experiments"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    kind: str | None = None,
):
    svc = service(request, "experiments")
    return await svc.list(limit=limit, kind=kind)


@router.get("/{experiment_id}", response_model=ExperimentSummary)
async def get_experiment(experiment_id: str, request: Request):
    svc = service(request, "experiments")
    result = await svc.get(experiment_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "EXPERIMENT_NOT_FOUND",
                "message": "Experiment was not found.",
            },
        )
    return result
