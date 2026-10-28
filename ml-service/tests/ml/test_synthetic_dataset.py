"""Tests for deterministic synthetic dataset generation."""

from ml.dataset.synthetic import generate_http_samples


def test_generator_is_reproducible():
    first = generate_http_samples(
        rows_per_class=2,
        seed=7,
    )
    second = generate_http_samples(
        rows_per_class=2,
        seed=7,
    )
    assert first.equals(second)
    assert len(first) == 10


def test_generator_has_all_five_classes():
    frame = generate_http_samples(rows_per_class=2)
    assert frame["label"].nunique() == 5
