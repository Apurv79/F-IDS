"""Smoke test for the Logistic Regression pipeline."""

from ml.dataset.synthetic import generate_http_samples
from ml.features.extractor import extract_frame
from ml.models.pipeline import build_training_pipeline


def test_logistic_regression_pipeline_fits():
    frame = generate_http_samples(
        rows_per_class=5,
        seed=42,
    )
    X = extract_frame(frame)
    y = frame["label"]
    pipeline = build_training_pipeline(
        "logistic_regression",
        random_seed=42,
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(frame)
