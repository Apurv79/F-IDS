"""AI explanation and summarization endpoints."""

from fastapi import APIRouter, Depends

from ai.explanation.service import ExplanationService
from ai.summarization.service import SummarizationService
from api.dependencies.auth import verify_internal_api_key
from shared.schemas.explanation import ExplanationRequest, ExplanationResponse
from shared.schemas.summarization import (
    SummarizationRequest,
    SummarizationResponse,
)

router = APIRouter(
    prefix="/api/v1/ai",
    tags=["AI Services"],
    dependencies=[Depends(verify_internal_api_key)],
)

_explanation_service = ExplanationService()
_summarization_service = SummarizationService()


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Generate AI Explanation & Defensive Guidance",
)
async def explain_detection(request: ExplanationRequest) -> ExplanationResponse:
    """Explain an attack classification for analysts without overriding predictions."""
    return await _explanation_service.explain(request)


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Generate AI Security Log Summary",
)
async def summarize_logs(request: SummarizationRequest) -> SummarizationResponse:
    """Summarize security logs and detect key threat observations."""
    return await _summarization_service.summarize(request)
