"""Train an FL-IDS classifier from a prepared CSV dataset.

Run from the ml-service root:
uv run python -m ml.training.train \
    --csv path/to/dataset.csv \
    --label-column label
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ml.dataset.loader import load_csv
from ml.evaluation.metrics import evaluate_classifier
from ml.evaluation.report import save_metrics_report
from ml.features.extractor import extract_frame
from ml.features.schema import FEATURE_SCHEMA_VERSION
from ml.models.artifacts import save_model
from ml.models.metadata import write_model_metadata
from ml.models.pipeline import build_training_pipeline
from ml.training.splits import split_dataset


def train_from_csv(
    csv_path: str,
    label_column: str = "label",
    algorithm: str = "logistic_regression",
    test_size: float = 0.2,
    random_seed: int = 42,
    dataset_version: str = "local-unversioned",
    output_dir: str = "artifacts/models",
) -> dict:
    """Train, evaluate, and persist an ML model from a CSV file."""
    frame = load_csv(csv_path, label_column=label_column)
    X = extract_frame(frame)
    y = frame[label_column]

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_seed=random_seed,
    )

    pipeline = build_training_pipeline(
        algorithm=algorithm,
        random_seed=random_seed,
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate_classifier(pipeline, X_test, y_test)

    # Save artifacts
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_version = f"{algorithm}_v_{timestamp}"
    save_dir = Path(output_dir) / model_version
    save_dir.mkdir(parents=True, exist_ok=True)

    artifact_file = save_dir / "model.joblib"
    save_model(pipeline, artifact_file)

    metadata_file = save_dir / "metadata.json"
    metadata = write_model_metadata(
        metadata_file,
        model_version=model_version,
        algorithm=algorithm,
        dataset_version=dataset_version,
        split_strategy=f"train_test_{int((1-test_size)*100)}_{int(test_size*100)}",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        metrics=metrics,
        artifact_path=str(artifact_file),
        random_seed=random_seed,
    )

    report_file = save_dir / "metrics.json"
    save_metrics_report(metrics, report_file)

    return {
        "model_version": model_version,
        "artifact_path": str(artifact_file),
        "metrics": metrics,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Train an FL-IDS classifier")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--label-column", default="label", help="Target label column name")
    parser.add_argument(
        "--algorithm",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "mlp"],
        help="Classification algorithm",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-version", default="local-unversioned", help="Dataset version identifier")
    parser.add_argument("--output-dir", default="artifacts/models", help="Directory for model output")

    args = parser.parse_args()

    result = train_from_csv(
        csv_path=args.csv,
        label_column=args.label_column,
        algorithm=args.algorithm,
        test_size=args.test_size,
        random_seed=args.seed,
        dataset_version=args.dataset_version,
        output_dir=args.output_dir,
    )

    print(f"Training completed successfully for version: {result['model_version']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"F1 Macro: {result['metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
