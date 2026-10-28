"""Authentication and validation dependencies for API endpoints."""

from api.dependencies.auth import verify_internal_api_key

__all__ = ["verify_internal_api_key"]
