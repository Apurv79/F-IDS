"""Tests for canonical dataset labels."""

import pytest

from ml.dataset.labels import normalize_label


def test_known_labels_are_normalized():
    assert normalize_label("benign") == "Normal"
    assert normalize_label("sqli") == "SQL Injection"
    assert normalize_label("xss") == "XSS"
    assert normalize_label("brute force") == "Brute Force"
    assert normalize_label("ddos") == "Other/Anomaly"


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        normalize_label("unknown class")
