"""Dataset integrity checks."""

from collections.abc import Sequence

import pandas as pd


def validate_dataframe(
    frame: pd.DataFrame,
    label_column: str = "label",
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate that a DataFrame has required columns, labels, and no empty rows."""
    if frame is None or frame.empty:
        raise ValueError("Dataset is empty")
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if required_columns:
        missing = set(required_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
    result = frame.copy()
    # Rows without labels cannot be used for supervised classification.
    result = result.dropna(subset=[label_column])
    if result.empty:
        raise ValueError("No rows remain after removing missing labels")
    result[label_column] = result[label_column].astype(str).str.strip()
    if (result[label_column] == "").any():
        raise ValueError("Dataset contains empty labels")
    # Remove exact duplicate rows.
    result = result.drop_duplicates().reset_index(drop=True)
    return result


def dataset_summary(
    frame: pd.DataFrame,
    label_column: str = "label",
) -> dict:
    """Return dataset size, missing-value count, and class distribution."""
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    counts = frame[label_column].value_counts(dropna=False).to_dict()
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {
            str(label): int(count)
            for label, count in counts.items()
        },
    }
