"""Helpers for converting Pydantic objects into JSON-compatible data."""

from typing import Any

from pydantic import BaseModel


def to_json_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a Pydantic model to a Python dictionary containing

    JSON-compatible values.

    Example:
        payload = to_json_dict(prediction_response)
    """
    return model.model_dump(mode="json")
