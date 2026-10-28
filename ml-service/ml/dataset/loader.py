"""Load and prepare local CSV datasets."""

from pathlib import Path

import pandas as pd

from ml.dataset.labels import normalize_label
from ml.dataset.validation import validate_dataframe


def load_csv(
    path: str | Path,
    label_column: str = "label",
) -> pd.DataFrame:
    """Load a CSV and normalize its target labels."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )
    frame = pd.read_csv(dataset_path, low_memory=False)
    if label_column not in frame.columns:
        raise ValueError(
            f"CSV must contain label column {label_column!r}"
        )
    # Map source labels into the five FL-IDS classes.
    frame[label_column] = frame[label_column].map(normalize_label)
    return validate_dataframe(
        frame,
        label_column=label_column,
    )
