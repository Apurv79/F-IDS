"""Preprocessing steps that are fitted on training data only."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    """Build a numeric preprocessing pipeline.

    The median imputer handles missing numeric values.
    StandardScaler normalizes feature magnitudes for algorithms
    such as Logistic Regression and MLP.
    Keeping preprocessing inside the model pipeline ensures
    inference uses the same fitted transformations as training.
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])
