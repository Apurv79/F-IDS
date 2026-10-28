from fastapi import APIRouter
from api.routes import (
    comparisons,
    datasets,
    experiments,
    federated,
    health,
    intelligence,
    models,
    training,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(intelligence.router)
api_router.include_router(training.router)
api_router.include_router(models.router)
api_router.include_router(federated.router)
api_router.include_router(experiments.router)
api_router.include_router(datasets.router)
api_router.include_router(comparisons.router)
