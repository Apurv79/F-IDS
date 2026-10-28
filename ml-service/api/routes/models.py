from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from api.contracts import (
    ModelDownloadResponse,
    ModelMetadata,
)
from api.dependencies import require_internal_api_key, service

router = APIRouter(
    prefix="/models",
    tags=["Models"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.get("", response_model=list[ModelMetadata])
async def list_models(request: Request):
    svc = service(request, "models")
    return await svc.list()


@router.get("/latest", response_model=ModelMetadata)
async def get_latest_model(request: Request):
    svc = service(request, "models")
    result = await svc.get_latest()
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_FOUND",
                "message": "No active model is available.",
            },
        )
    return result


@router.get("/{model_version}", response_model=ModelMetadata)
async def get_model(model_version: str, request: Request):
    svc = service(request, "models")
    result = await svc.get(model_version)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_FOUND",
                "message": "Model version was not found.",
            },
        )
    return result


@router.post(
    "/{model_version}/download",
    response_model=ModelDownloadResponse,
    summary="Create a short-lived model download URL",
)
async def get_model_download(model_version: str, request: Request):
    svc = service(request, "models")
    if not hasattr(svc, "create_download_url"):
        raise HTTPException(
            status_code=501,
            detail={
                "code": "OPERATION_NOT_IMPLEMENTED",
                "message": (
                    "Model store must implement "
                    "create_download_url(version)."
                ),
            },
        )
    result = await svc.create_download_url(model_version)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_ARTIFACT_NOT_FOUND",
                "message": "Model artifact was not found.",
            },
        )
    return result
