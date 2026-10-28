from fastapi import APIRouter, Request
from api.contracts import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
async def live(request: Request):
    return HealthResponse(
        status="ok",
        version=request.app.version,
        checks={"process": "ok"},
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
)
async def ready(request: Request):
    services = getattr(request.app.state, "services", {})
    checks = {
        name: ("ok" if value is not None else "not_configured")
        for name, value in services.items()
    }
    required = (
        "prediction",
        "training",
        "federated",
        "explanation",
        "summarization",
        "models",
        "datasets",
        "experiments",
    )
    ready_state = all(checks.get(name) == "ok" for name in required)
    return HealthResponse(
        status="ok" if ready_state else "not_ready",
        version=request.app.version,
        checks=checks,
    )
