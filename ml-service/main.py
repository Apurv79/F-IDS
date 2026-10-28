"""FastAPI entry point for the FL-IDS intelligence service."""

import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.adapters import (
    DatasetsAdapter,
    ExperimentsAdapter,
    ExplanationAdapter,
    FederatedAdapter,
    ModelsAdapter,
    PredictionAdapter,
    SummarizationAdapter,
    TrainingAdapter,
)
from api.router import api_router

SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize service state and register domain service adapters
    app.state.internal_api_key = os.getenv("INTERNAL_API_KEY", "")

    app.state.services = {
        "prediction": PredictionAdapter(),
        "explanation": ExplanationAdapter(),
        "summarization": SummarizationAdapter(),
        "training": TrainingAdapter(),
        "models": ModelsAdapter(),
        "federated": FederatedAdapter(),
        "datasets": DatasetsAdapter(),
        "experiments": ExperimentsAdapter(),
    }
    print("FL-IDS Intelligence Service started with all adapters registered.")

    yield

    print("FL-IDS Intelligence Service shutting down...")


app = FastAPI(
    title="FL-IDS Intelligence Service API",
    version=SERVICE_VERSION,
    description=(
        "Production REST interface between the Node.js Backend and "
        "the Python Machine Learning, AI Explanation, and Federated Learning service."
    ),
    lifespan=lifespan,
)

# Initialize state on application instance
app.state.internal_api_key = os.getenv("INTERNAL_API_KEY", "")
app.state.services = {
    "prediction": PredictionAdapter(),
    "explanation": ExplanationAdapter(),
    "summarization": SummarizationAdapter(),
    "training": TrainingAdapter(),
    "models": ModelsAdapter(),
    "federated": FederatedAdapter(),
    "datasets": DatasetsAdapter(),
    "experiments": ExperimentsAdapter(),
}

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_correlation_and_tracing(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request payload validation failed.",
                "details": jsonable_encoder(exc.errors()),
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", "HTTP_ERROR")
        message = detail.get("message", "An error occurred.")
    else:
        code = "HTTP_ERROR"
        message = str(detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


# Mount the versioned API router
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)