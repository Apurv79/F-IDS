"""Comprehensive testing for multi-round Federated Learning and edge cases."""

import numpy as np
import pytest

from federated.aggregation.fedavg import aggregate_parameters
from federated.rounds.coordinator import FederatedCoordinator
from ml.dataset.synthetic import generate_http_samples


def test_mismatched_parameter_shapes_rejected():
    updates = [
        ([[1.0, 2.0], [0.5]], 10),
        ([[1.0, 2.0, 3.0], [0.5]], 10),  # Mismatched column count
    ]
    with pytest.raises(ValueError, match="shapes do not match"):
        aggregate_parameters(updates)


def test_non_positive_example_count_rejected():
    updates = [
        ([[1.0, 2.0], [0.5]], 0),
    ]
    with pytest.raises(ValueError, match="must be positive"):
        aggregate_parameters(updates)


def test_consecutive_rounds_global_progression():
    dataset = generate_http_samples(rows_per_class=20, seed=42)
    coordinator = FederatedCoordinator(initial_model_version="v1.0.0")

    # Round 1
    res_r1 = coordinator.run_simulation(dataset, num_clients=3, partition_strategy="iid")
    assert res_r1["previous_model_version"] == "v1.0.0"
    assert res_r1["new_model_version"] == "v1.0.0-next"
    assert coordinator.global_parameters is not None

    # Round 2
    res_r2 = coordinator.run_simulation(dataset, num_clients=3, partition_strategy="non_iid")
    assert res_r2["previous_model_version"] == "v1.0.0-next"
    assert res_r2["new_model_version"] == "v1.0.0-next-next"
    assert res_r2["test_accuracy"] >= 0.8
