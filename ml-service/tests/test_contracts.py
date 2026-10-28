import pytest
from pydantic import ValidationError

from api.contracts import (
    FederatedRoundCreate,
    FederatedUpdateRequest,
    PredictionRequest,
    SummarizeRequest,
)


def test_prediction_requires_exactly_one_input_mode():
    with pytest.raises(ValidationError):
        PredictionRequest()
    with pytest.raises(ValidationError):
        PredictionRequest(
            request={"method": "GET", "url": "/"},
            features={"url_length": 1},
        )
    payload = PredictionRequest(
        request={"method": "GET", "url": "/health"}
    )
    assert payload.request.method == "GET"


def test_federated_round_validates_min_clients():
    with pytest.raises(ValidationError):
        FederatedRoundCreate(
            num_clients=3,
            min_clients=4,
        )


def test_federated_updates_reject_non_finite_parameters():
    with pytest.raises(ValidationError):
        FederatedUpdateRequest(
            client_id="c1",
            round_id="r1",
            base_model_version="v1",
            num_examples=3,
            parameters=[[float("nan")]],
        )


def test_summary_time_window_validation():
    with pytest.raises(ValidationError):
        SummarizeRequest(
            logs=["event"],
            time_window_start="2026-10-29T12:00:00Z",
            time_window_end="2026-10-29T11:00:00Z",
        )
