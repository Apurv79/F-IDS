"""Integration tests for Phase 5 REST API routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.mark.asyncio
async def test_api_health_live_and_ready():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_live = await client.get("/api/v1/health/live")
        assert res_live.status_code == 200
        assert res_live.json()["status"] == "ok"

        res_ready = await client.get("/api/v1/health/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predictions_endpoint_sql_injection():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "event_id": "test_evt_101",
            "request": {
                "method": "GET",
                "url": "/products?id=1%20OR%201=1",
                "headers": {"user-agent": "sqlmap/1.5"},
                "body": "",
                "request_rate": 25.0,
                "failed_login_count": 0,
            },
        }
        response = await client.post("/api/v1/predictions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "SQL Injection"
        assert data["confidence"] > 0.5
        assert data["event_id"] == "test_evt_101"
        assert "class_probabilities" in data


@pytest.mark.asyncio
async def test_federated_round_lifecycle():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create round
        res_create = await client.post("/api/v1/federated/rounds", json={"num_clients": 3, "min_clients": 2})
        assert res_create.status_code == 201
        round_id = res_create.json()["round_id"]

        # Submit update
        update_payload = {
            "client_id": "client_1",
            "round_id": round_id,
            "base_model_version": "v1.0.0",
            "num_examples": 100,
            "parameters": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                [0.1],
            ],
        }
        res_update = await client.post(f"/api/v1/federated/rounds/{round_id}/updates", json=update_payload)
        assert res_update.status_code == 202

        # Aggregate round
        res_agg = await client.post(f"/api/v1/federated/rounds/{round_id}/aggregate", json={"force": True})
        assert res_agg.status_code == 200
        assert res_agg.json()["status"] == "completed"
