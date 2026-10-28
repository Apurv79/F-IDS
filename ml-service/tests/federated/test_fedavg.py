"""Tests for weighted FedAvg aggregation."""

import pytest

from federated.aggregation.fedavg import aggregate_parameters


def test_weighted_average():
    updates = [
        ([[1.0, 3.0], [2.0]], 1),
        ([[3.0, 5.0], [4.0]], 3),
    ]
    result = aggregate_parameters(updates)
    assert result[0] == pytest.approx([2.5, 4.5])
    assert result[1] == pytest.approx([3.5])


def test_empty_updates_are_rejected():
    with pytest.raises(ValueError):
        aggregate_parameters([])
