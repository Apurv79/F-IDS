"""Tests for HTTP feature extraction."""

from ml.features.extractor import (
    FEATURE_COLUMNS,
    extract_features,
    shannon_entropy,
)


def test_features_are_numeric_and_ordered():
    result = extract_features({
        "url": "/search?q=%3Cscript%3Ealert(1)%3C/script%3E",
        "method": "GET",
        "headers": {"User-Agent": "Mozilla/5.0"},
        "body": "",
    })
    assert list(result.keys()) == FEATURE_COLUMNS
    assert all(
        isinstance(value, float)
        for value in result.values()
    )


def test_script_pattern_is_detected():
    result = extract_features({
        "url": "/search?q=%3Cscript%3Ealert(1)%3C/script%3E",
        "method": "GET",
        "body": "",
    })
    assert result["has_script_tag"] == 1.0


def test_empty_entropy_is_zero():
    assert shannon_entropy("") == 0.0
