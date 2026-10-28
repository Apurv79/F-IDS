"""API route blueprints for FL-IDS Intelligence Service."""

from api.routes.ai import router as ai_router
from api.routes.federated import router as federated_router
from api.routes.prediction import router as prediction_router

__all__ = ["ai_router", "federated_router", "prediction_router"]
