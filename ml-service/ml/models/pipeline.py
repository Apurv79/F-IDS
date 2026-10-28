"""Construct the full trainable model pipeline."""

from sklearn.pipeline import Pipeline

from ml.features.preprocessing import build_preprocessor
from ml.models.factory import create_model


def build_training_pipeline(
    algorithm: str = "logistic_regression",
    random_seed: int = 42,
) -> Pipeline:
    """Create preprocessing + classifier as a single pipeline.

    The fitted pipeline can be saved as one artifact, preventing
    training and inference from using different preprocessing.
    """
    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(),
        ),
        (
            "classifier",
            create_model(algorithm, random_seed),
        ),
    ])
