"""Metadata for reproducible model artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def write_model_metadata(
    output_path: str | Path,
    *,
    model_version: str,
    algorithm: str,
    dataset_version: str,
    split_strategy: str,
    feature_schema_version: str,
    metrics: dict[str, Any],
    artifact_path: str,
    random_seed: int,
) -> dict:
    """Write model metadata beside the trained artifact.

    Production application-facing model metadata will ultimately
    be maintained by the main backend database.
    """
    metadata = {
        "model_version": model_version,
        "algorithm": algorithm,
        "dataset_version": dataset_version,
        "split_strategy": split_strategy,
        "feature_schema_version": feature_schema_version,
        "metrics": metrics,
        "artifact_path": artifact_path,
        "random_seed": random_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata
