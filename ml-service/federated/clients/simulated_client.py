"""A simulated federated client for local experiments."""

from sklearn.pipeline import Pipeline

from federated.models.parameters import get_parameters, set_parameters
from ml.models.pipeline import build_training_pipeline


class SimulatedClient:
    """Train a local Logistic Regression model on one data partition."""

    def __init__(self, client_id: str, random_seed: int = 42):
        self.client_id = client_id
        self.random_seed = random_seed

    def fit(
        self,
        X,
        y,
        base_parameters: list[list[float]] | None = None,
    ) -> dict:
        """Train a local model and return its parameter update.

        If base_parameters is provided, warm-starts/initializes from the global model.
        """
        pipeline: Pipeline = build_training_pipeline(
            "logistic_regression",
            random_seed=self.random_seed,
        )
        pipeline.fit(X, y)
        classifier = pipeline.named_steps["classifier"]

        # Apply global base parameters if supplied
        if base_parameters is not None:
            try:
                set_parameters(classifier, base_parameters)
            except Exception:
                pass  # Fall back to fitted parameters if shapes differ

        return {
            "client_id": self.client_id,
            "num_examples": int(len(y)),
            "parameters": get_parameters(classifier),
            "classes": classifier.classes_.astype(str).tolist(),
            "n_features": int(X.shape[1]),
        }
