# Export the base exception and specific exceptions so other
# modules can import them directly from shared.exceptions.
from shared.exceptions.base import (
    AIProviderError,
    DatasetValidationError,
    FLIDSException,
    InvalidFeaturesError,
    InvalidFederatedUpdateError,
    ModelNotLoadedError,
)

__all__ = [
    "FLIDSException",
    "InvalidFeaturesError",
    "ModelNotLoadedError",
    "DatasetValidationError",
    "AIProviderError",
    "InvalidFederatedUpdateError",
]
