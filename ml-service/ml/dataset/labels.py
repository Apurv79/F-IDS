"""Canonical class labels used throughout FL-IDS."""

CANONICAL_LABELS = (
    "Normal",
    "SQL Injection",
    "XSS",
    "Brute Force",
    "Other/Anomaly",
)

# Map common labels from external datasets to our project labels.
_LABEL_ALIASES = {
    "normal": "Normal",
    "benign": "Normal",
    "legitimate": "Normal",
    "sql injection": "SQL Injection",
    "sql_injection": "SQL Injection",
    "sqli": "SQL Injection",
    "xss": "XSS",
    "cross site scripting": "XSS",
    "cross-site scripting": "XSS",
    "brute force": "Brute Force",
    "brute_force": "Brute Force",
    "login brute force": "Brute Force",
    "anomaly": "Other/Anomaly",
    "other": "Other/Anomaly",
    "other/anomaly": "Other/Anomaly",
    "ddos": "Other/Anomaly",
    "dos": "Other/Anomaly",
}


def normalize_label(value: str) -> str:
    """Convert a recognized source label to its canonical FL-IDS label."""
    if value is None:
        raise ValueError("Label cannot be empty")
    key = str(value).strip().lower()
    label = _LABEL_ALIASES.get(key)
    if label is None:
        # Fail explicitly rather than silently assigning the wrong class.
        raise ValueError(
            f"Unsupported label {value!r}. "
            "Add a reviewed mapping for this source label."
        )
    return label
