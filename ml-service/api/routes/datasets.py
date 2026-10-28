from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
)
from api.contracts import (
    DatasetMetadata,
    DatasetValidationRequest,
    DatasetValidationResponse,
)
from api.dependencies import require_internal_api_key, service

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.get("", response_model=list[DatasetMetadata])
async def list_datasets(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    svc = service(request, "datasets")
    return await svc.list(limit=limit)


@router.get("/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset(dataset_id: str, request: Request):
    svc = service(request, "datasets")
    result = await svc.get(dataset_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "DATASET_NOT_FOUND",
                "message": "Dataset was not found.",
            },
        )
    return result


@router.post(
    "/validate",
    response_model=DatasetValidationResponse,
    summary="Validate a registered dataset before training",
)
async def validate_dataset(
    payload: DatasetValidationRequest,
    request: Request,
):
    svc = service(request, "datasets")
    return await svc.validate(payload)
