"""Internal service-to-service authentication and adapter resolution."""

from __future__ import annotations
import secrets
from typing import Any
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

internal_key_header = APIKeyHeader(
    name="X-Internal-API-Key",
    auto_error=False,
    scheme_name="InternalServiceApiKey",
)


async def require_internal_api_key(
    request: Request,
    x_internal_api_key: str | None = Security(internal_key_header),
) -> None:
    """Authenticate backend-to-ML-service requests."""
    expected = getattr(request.app.state, "internal_api_key", "")
    # If no key configured in environment, allow local development
    if not expected:
        return

    if (
        not x_internal_api_key
        or not secrets.compare_digest(x_internal_api_key, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing internal service API key.",
            },
        )


def service(request: Request, name: str) -> Any:
    """Resolve a concrete service adapter registered on the app state."""
    services = getattr(request.app.state, "services", {})
    adapter = services.get(name)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_NOT_INITIALIZED",
                "message": f"Service adapter '{name}' is not configured.",
            },
        )
    return adapter
