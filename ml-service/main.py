"""
FL-IDS Intelligence Service.

FastAPI entry point for the ML/AI/Federated Learning modular monolith.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown."""

    # Startup: initialize resources when they are added.
    print("FL-IDS ML service starting...")

    yield

    # Shutdown: clean up resources when they are added.
    print("FL-IDS ML service shutting down...")


# Create the main FastAPI application.
app = FastAPI(
    title="FL-IDS Intelligence Service",
    description=(
        "Machine Learning, Artificial Intelligence, and "
        "Federated Learning service for the FL-IDS project."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# Enable CORS for local development.
# Production origins will come from configuration later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    tags=["Health"],
    summary="Application health check",
)
async def health_check() -> dict[str, str]:
    """Check whether the application is running."""

    return {
        "status": "healthy",
        "service": "ml-service",
        "version": "0.1.0",
    }


@app.get(
    "/ready",
    tags=["Health"],
    summary="Application readiness check",
)
async def readiness_check() -> dict[str, str]:
    """Check whether the service is ready to accept work."""

    return {
        "status": "ready",
        "service": "ml-service",
    }


@app.get(
    "/",
    tags=["System"],
    summary="Service information",
)
async def root() -> dict[str, str]:
    """Return basic service information."""

    return {
        "service": "FL-IDS Intelligence Service",
        "version": "0.1.0",
        "status": "running",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected application errors safely."""

    # Log the real error internally; don't expose implementation details.
    print(
        f"Unhandled exception on "
        f"{request.method} {request.url.path}: {exc}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal error occurred.",
            }
        },
    )


from api.routes import ai_router, federated_router, prediction_router

# Register API routes
app.include_router(prediction_router)
app.include_router(ai_router)
app.include_router(federated_router)


if __name__ == "__main__":
    # Run the FastAPI application locally.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Development only; disable in production.
    )