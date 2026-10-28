from fastapi import APIRouter, Depends, Request

from api.contracts import (
    ExplainRequest,
    ExplainResponse,
    PredictionRequest,
    PredictionResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from api.dependencies import require_internal_api_key, service

router = APIRouter(
    tags=["Intelligence"],
    dependencies=[Depends(require_internal_api_key)],
)


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Classify a web request",
)
async def predict(payload: PredictionRequest, request: Request):
    svc = service(request, "prediction")
    if hasattr(svc, "predict"):
        result = await svc.predict(payload)
    else:
        result = await svc(payload)
    return result


@router.post(
    "/ai/explanations",
    response_model=ExplainResponse,
    summary="Explain a detected attack",
)
async def explain(payload: ExplainRequest, request: Request):
    svc = service(request, "explanation")
    if hasattr(svc, "explain"):
        result = await svc.explain(payload)
    else:
        result = await svc(payload)
    return result


@router.post(
    "/ai/summaries",
    response_model=SummarizeResponse,
    summary="Summarize attack logs",
)
async def summarize(payload: SummarizeRequest, request: Request):
    svc = service(request, "summarization")
    if hasattr(svc, "summarize"):
        result = await svc.summarize(payload)
    else:
        result = await svc(payload)
    return result
