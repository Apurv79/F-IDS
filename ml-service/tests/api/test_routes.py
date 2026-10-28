"""Integration tests for Prediction, AI, and Federated API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.mark.asyncio
async def test_predict_raw_request_sql_injection():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "event_id": "test_evt_101",
            "raw_request": {
                "url": "/products?id=1%20OR%201=1",
                "method": "GET",
                "headers": {"user-agent": "sqlmap/1.5"},
                "body": "",
                "ip_request_rate": 25.0,
                "failed_login_count": 0.0,
            },
        }
        response = await client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "SQL Injection"
        assert data["confidence"] > 0.5
        assert data["event_id"] == "test_evt_101"
        assert "request_id" in data
        assert "class_probabilities" in data


@pytest.mark.asyncio
async def test_predict_raw_request_normal():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "raw_request": {
                "url": "/home",
                "method": "GET",
                "headers": {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                "body": "",
                "ip_request_rate": 2.0,
                "failed_login_count": 0.0,
            },
        }
        response = await client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Normal"


@pytest.mark.asyncio
async def test_federated_simulate_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "num_clients": 3,
            "partition_strategy": "iid",
            "sample_size_per_class": 15,
        }
        response = await client.post("/api/v1/federated/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["num_clients"] == 3
        assert "round_id" in data
        assert data["test_accuracy"] >= 0.8
