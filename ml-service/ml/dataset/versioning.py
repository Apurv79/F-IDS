"""Dataset versioning and integrity metadata."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    # Process chunks so large datasets don't need to fit in memory.
    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_manifest(
    dataset_path: str | Path,
    dataset_version: str,
    output_path: str | Path,
    label_column: str = "label",
) -> dict:
    """Write metadata that identifies a particular dataset version."""
    source = Path(dataset_path)
    manifest = {
        "dataset_version": dataset_version,
        "file_name": source.name,
        "sha256": sha256_file(source),
        "label_column": label_column,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
