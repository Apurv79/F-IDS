"""Experiment tracking for reproducible ML comparisons."""

from datetime import datetime, timezone
import json
from pathlib import Path


def write_experiment_record(
    output_path: str | Path,
    record: dict,
) -> None:
    """Save an experiment record.

    Include algorithm, dataset version, split strategy,
    random seed, model version, metrics, and FL round details
    when those details are available.
    """
    payload = dict(record)
    payload.setdefault(
        "created_at",
        datetime.now(timezone.utc).isoformat(),
    )
    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
