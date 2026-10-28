"""Save evaluation metrics to a JSON report."""

import json
from pathlib import Path


def save_metrics_report(
    metrics: dict,
    path: str | Path,
) -> Path:
    """Write metrics and per-class results to disk."""
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return destination
