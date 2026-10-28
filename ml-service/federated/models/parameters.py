"""Extract and apply parameters for compatible federated models."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def get_parameters(model: LogisticRegression) -> list[list[float]]:
    """Export model coefficients and intercepts as JSON-friendly arrays."""
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("Model must be fitted before exporting parameters")
    return [
        model.coef_.astype(float).tolist(),
        model.intercept_.astype(float).tolist(),
    ]


def validate_parameters(parameters: list[list[float]]) -> None:
    """Reject malformed, empty, or non-finite parameter arrays."""
    if len(parameters) != 2:
        raise ValueError("Expected coefficient and intercept arrays")
    coefficients = np.asarray(parameters[0], dtype=float)
    intercepts = np.asarray(parameters[1], dtype=float)
    if coefficients.size == 0 or intercepts.size == 0:
        raise ValueError("Parameter arrays cannot be empty")
    if not np.isfinite(coefficients).all():
        raise ValueError("Coefficients contain non-finite values")
    if not np.isfinite(intercepts).all():
        raise ValueError("Intercepts contain non-finite values")


def set_parameters(model: LogisticRegression, parameters: list[list[float]]) -> None:
    """Apply aggregated parameter arrays directly to a fitted model."""
    validate_parameters(parameters)
    model.coef_ = np.asarray(parameters[0], dtype=float)
    model.intercept_ = np.asarray(parameters[1], dtype=float)
