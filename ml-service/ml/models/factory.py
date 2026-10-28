"""Factory for supported FL-IDS classification algorithms."""

from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Logistic Regression is the baseline.
# Random Forest is the comparison model.
# MLP is optional for later experiments.
Algorithm = Literal[
    "logistic_regression",
    "random_forest",
    "mlp",
]


def create_model(
    algorithm: Algorithm = "logistic_regression",
    random_seed: int = 42,
):
    """Create a fresh classifier with reproducible settings."""
    if algorithm == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_seed,
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_seed,
        )
    if algorithm == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")
