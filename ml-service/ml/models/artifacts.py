"""Local model artifact persistence."""

from pathlib import Path

import joblib


def save_model(model, path: str | Path) -> Path:
    """Save a fitted model pipeline to disk."""
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(model, destination)
    return destination


def load_model(path: str | Path):
    """Load a trusted model artifact from disk."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {source}"
        )
    # Only load artifacts created by this project or another
    # trusted source. Joblib files can execute code during loading.
    return joblib.load(source)
