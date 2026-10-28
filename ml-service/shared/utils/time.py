"""Helpers for consistent UTC timestamps."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return utc_now().isoformat()
