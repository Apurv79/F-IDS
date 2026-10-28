"""Application-specific exceptions for FL-IDS.

These exceptions describe expected failure categories.
API routes will later translate them into safe HTTP responses.
"""


class FLIDSException(Exception):
    """Base class for errors raised by the FL-IDS service."""

    def __init__(self, message: str, code: str = "FLIDS_ERROR"):
        # Human-readable internal message for logs and debugging.
        self.message = message
        # Stable machine-readable code for error handling.
        self.code = code
        super().__init__(message)


class InvalidFeaturesError(FLIDSException):
    """Raised when input features are missing or invalid."""

    def __init__(self, message: str = "Invalid input features"):
        super().__init__(message, code="INVALID_FEATURES")


class ModelNotLoadedError(FLIDSException):
    """Raised when inference is requested without a loaded model."""

    def __init__(self, message: str = "ML model is not loaded"):
        super().__init__(message, code="MODEL_NOT_LOADED")


class DatasetValidationError(FLIDSException):
    """Raised when a dataset fails validation."""

    def __init__(self, message: str = "Dataset validation failed"):
        super().__init__(message, code="DATASET_VALIDATION_ERROR")


class AIProviderError(FLIDSException):
    """Raised when an AI provider fails or returns an unusable response."""

    def __init__(self, message: str = "AI provider request failed"):
        super().__init__(message, code="AI_PROVIDER_ERROR")


class InvalidFederatedUpdateError(FLIDSException):
    """Raised when a federated client update fails validation."""

    def __init__(self, message: str = "Invalid federated update"):
        super().__init__(message, code="INVALID_FEDERATED_UPDATE")
