"""Security dependencies for FastAPI routes."""

from fastapi import Header, HTTPException, status
from shared.config import get_settings


async def verify_internal_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Validate internal communication between Node.js and FastAPI.

    If INTERNAL_API_KEY is configured in .env, require X-API-Key header.
    If INTERNAL_API_KEY is empty (local dev convenience), bypass check.
    """
    settings = get_settings()
    if settings.internal_api_key and x_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing internal API key.",
            },
        )
