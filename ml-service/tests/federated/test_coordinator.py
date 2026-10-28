"""Tests for federated round version checks."""

import pytest

from federated.rounds.coordinator import FederatedCoordinator


def test_stale_update_is_rejected():
    coordinator = FederatedCoordinator("v1")
    coordinator.start_round()
    with pytest.raises(ValueError, match="Stale"):
        coordinator.submit_update(
            client_id="client-1",
            base_model_version="old-version",
            parameters=[[1.0, 2.0], [3.0]],
            num_examples=10,
        )


def test_valid_updates_can_be_aggregated():
    coordinator = FederatedCoordinator("v1")
    coordinator.start_round()
    coordinator.submit_update(
        client_id="client-1",
        base_model_version="v1",
        parameters=[[1.0, 2.0], [3.0]],
        num_examples=10,
    )
    result = coordinator.aggregate()
    assert result["clients_aggregated"] == 1
    assert result["previous_model_version"] == "v1"
    assert result["new_model_version"] == "v1-next"
