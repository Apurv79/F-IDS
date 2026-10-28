from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from api.contracts import (
    JobAccepted,
    JobStatusResponse,
    TrainingRequest,
)
from api.dependencies import require_internal_api_key, service

router = APIRouter(
    prefix="/training",
    tags=["Training"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.post(
    "/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue centralized training",
)
async def create_training_job(
    payload: TrainingRequest,
    request: Request,
):
    svc = service(request, "training")
    if hasattr(svc, "submit"):
        return await svc.submit(payload)
    return await svc(payload)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Check status of a training job",
)
async def get_training_job(job_id: str, request: Request):
    svc = service(request, "training")
    result = await svc.get_job(job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Training job '{job_id}' was not found.",
            },
        )
    return result
