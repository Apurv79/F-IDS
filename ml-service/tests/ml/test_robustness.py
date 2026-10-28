"""Tests for model loading errors, corrupt artifacts, and malformed features."""

from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression

from ml.dataset.labels import normalize_label
from ml.dataset.validation import validate_dataframe
from ml.features.extractor import FEATURE_COLUMNS, extract_features
from ml.models.artifacts import load_model, save_model


def test_unseen_and_empty_label_rejection():
    with pytest.raises(ValueError, match="Unsupported label"):
        normalize_label("unknown_ransomware_variant")
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_label(None)


def test_missing_features_are_zero_filled():
    # Only providing one key; all other 11 must safely default to 0.0
    res = extract_features({"url": "/test"})
    assert set(res.keys()) == set(FEATURE_COLUMNS)
    assert res["url_length"] == 5.0
    assert res["failed_login_count"] == 0.0
    assert res["sql_keyword_score"] == 0.0


def test_load_nonexistent_model_artifact_raises():
    with pytest.raises(FileNotFoundError):
        load_model(Path("/tmp/nonexistent_model_fids_99999.joblib"))


def test_save_and_load_model_roundtrip(tmp_path):
    model = LogisticRegression()
    # Fit dummy data
    model.fit([[1.0, 2.0], [3.0, 4.0]], ["Normal", "XSS"])

    target_file = tmp_path / "model.joblib"
    save_model(model, target_file)
    assert target_file.exists()

    loaded = load_model(target_file)
    assert list(loaded.classes_) == ["Normal", "XSS"]
    assert loaded.predict([[1.0, 2.0]])[0] == "Normal"
