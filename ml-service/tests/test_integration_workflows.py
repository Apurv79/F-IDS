"""Cross-module end-to-end integration workflows (A through F)."""

import pytest
from httpx import ASGITransport, AsyncClient

from ai.explanation.service import ExplanationService
from ai.providers.manager import AIProviderManager
from federated.rounds.coordinator import FederatedCoordinator
from main import app
from ml.dataset.synthetic import generate_http_samples
from ml.features.extractor import FEATURE_COLUMNS, extract_features
from ml.models.pipeline import build_training_pipeline
from shared.schemas.explanation import ExplanationRequest


class MockProviderSuccess:
    async def generate(self, prompt: str) -> str:
        return "Detection confirms malicious payload injection. Recommendations:\n1. Sanitize user input.\n2. Review WAF rules."


class MockProviderFail:
    async def generate(self, prompt: str) -> str:
        raise RuntimeError("Service down")


# Workflow A: Prediction pipeline end-to-end
def test_workflow_a_prediction_pipeline():
    raw_event = {
        "url": "/login?admin=1%27%20OR%20%271%27=%271",
        "method": "POST",
        "headers": {"User-Agent": "sqlmap/1.4"},
        "body": "{\"user\":\"admin\"}",
        "ip_request_rate": 40.0,
        "failed_login_count": 6.0,
    }
    extracted = extract_features(raw_event)
    assert extracted["has_sql_operator_pattern"] == 1.0
    assert extracted["url_length"] > 10.0

    # Feed into model
    samples = generate_http_samples(rows_per_class=10, seed=42)
    pipeline = build_training_pipeline("logistic_regression", random_seed=42)
    from ml.features.extractor import extract_frame
    X = extract_frame(samples)
    y = samples["label"]
    pipeline.fit(X, y)

    import pandas as pd
    X_single = pd.DataFrame([extracted], columns=FEATURE_COLUMNS).fillna(0.0)
    pred = pipeline.predict(X_single)[0]
    assert pred in ["SQL Injection", "Brute Force", "Other/Anomaly", "Normal", "XSS"]


# Workflow C: Federated Learning Multi-Round Simulation
def test_workflow_c_federated_multi_round():
    dataset = generate_http_samples(rows_per_class=15, seed=42)
    coordinator = FederatedCoordinator(initial_model_version="v1.0.0")

    result = coordinator.run_simulation(dataset, num_clients=3, partition_strategy="iid")
    assert result["num_clients"] == 3
    assert result["new_model_version"] == "v1.0.0-next"
    assert result["test_accuracy"] >= 0.7


# Workflow D: AI Analysis with graceful degradation
@pytest.mark.asyncio
async def test_workflow_d_ai_analysis_degradation():
    failing_manager = AIProviderManager(
        primary=MockProviderFail(),
        fallback=MockProviderFail(),
    )
    service = ExplanationService(provider_manager=failing_manager)
    req = ExplanationRequest(
        prediction="SQL Injection",
        features={"url_length": 30.0},
        confidence=0.98,
        model_version="v1.0.0",
    )
    res = await service.explain(req)
    assert res.degraded is True
    assert res.provider == "unavailable"
    assert len(res.recommendations) > 0


# Workflow F: HTTP API Contract Enforcement
@pytest.mark.asyncio
async def test_workflow_f_api_contract_enforcement():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Invalid payload missing both request and features must return 422
        bad_response = await client.post("/api/v1/predictions", json={})
        assert bad_response.status_code == 422
        err = bad_response.json()
        assert "error" in err
        assert err["error"]["code"] == "VALIDATION_ERROR"
