# Backend <-> ML Service API Contract (F-IDS)

This document formalizes the exact HTTP REST interface contract between the **Node.js Express Backend** (`http://localhost:5000`) and the **FastAPI Machine Learning Service** (`http://localhost:8000`).

---

## 1. Architectural Roles & Boundaries

| Responsibility | Node.js Backend Service | FastAPI ML Service |
| :--- | :--- | :--- |
| **Ingress Interception** | Intercepts live incoming HTTP requests via middleware | Does not touch live client traffic directly |
| **Telemetry & State** | Tracks client IP request rates and failed logins in Redis | Stateless; receives metrics with request |
| **Feature Extraction** | Does NOT perform mathematical or regex feature extraction | Extracts 12 numerical features & Shannon entropy |
| **Inference** | Makes synchronous HTTP POST to `/api/v1/predict` | Runs model pipeline, returns predictions & probabilities |
| **Policy Enforcement** | Decides whether to block IP, throttle, or alert | Advisory; returns prediction and confidence only |
| **Alert Storage** | Persists incident records in MongoDB | Does not maintain long-term incident database |
| **AI Explanation** | Dispatches async BullMQ background job for explanations | Generates defensive analysis & incident summary |
| **Federated Learning** | Node cluster orchestration, node tokens | FedAvg aggregation, client weight validation, rounds |

---

## 2. Authentication & Security

- **Internal API Key Header**: All inter-service calls from Node.js to FastAPI must include:
  ```http
  X-API-Key: <INTERNAL_API_KEY>
  ```
  *(During local development, if `INTERNAL_API_KEY` is blank in `.env`, authentication is automatically bypassed).*
- **Failure Policy**:
  - **Synchronous Inference Timeout**: `500ms` (Axios client timeout).
  - **Fail-Open vs. Fail-Closed**: In detection/monitor mode (default), Node.js fails open (allows request through if ML service is down, but emits a `ML_SERVICE_UNAVAILABLE` warning log). In strict blocking mode, Node.js fails closed on critical admin/auth routes.

---

## 3. Endpoints

### 3.1 Real-Time Attack Prediction (`POST /api/v1/predict`)
Synchronous endpoint called by Node.js to classify incoming traffic.

#### Request Headers:
```http
Content-Type: application/json
X-API-Key: your_internal_secret_key
```

#### Request Payload:
Node.js passes the raw sanitized HTTP event directly:
```json
{
  "event_id": "evt_987654321",
  "raw_request": {
    "url": "/login?user=admin",
    "method": "POST",
    "headers": {
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    },
    "body": "{\"username\":\"admin\",\"password\":\"' OR 1=1 --\"}",
    "ip_request_rate": 35.0,
    "failed_login_count": 4.0
  }
}
```

#### Success Response (`200 OK`):
```json
{
  "prediction": "SQL Injection",
  "confidence": 0.985,
  "model_version": "logistic_regression_v1.0.0",
  "class_probabilities": {
    "Normal": 0.005,
    "SQL Injection": 0.985,
    "XSS": 0.003,
    "Brute Force": 0.005,
    "Other/Anomaly": 0.002
  },
  "event_id": "evt_987654321",
  "request_id": "c1f11c52-78a4-4a46-8162-8178d467be64"
}
```

#### Supported Canonical Attack Classes:
1. `Normal`
2. `SQL Injection`
3. `XSS`
4. `Brute Force`
5. `Other/Anomaly`

---

### 3.2 AI Attack Explanation (`POST /api/v1/ai/explain`)
Called asynchronously by background BullMQ workers when an alert is flagged.

#### Request Payload:
```json
{
  "prediction": "SQL Injection",
  "confidence": 0.985,
  "model_version": "logistic_regression_v1.0.0",
  "features": {
    "url_length": 25.0,
    "has_sql_operator_pattern": 1.0,
    "sql_keyword_score": 2.0,
    "payload_entropy": 3.8
  }
}
```

#### Success Response (`200 OK`):
```json
{
  "explanation": "The request contains common SQL injection syntax (' OR 1=1) targeting user authentication fields.",
  "recommendations": [
    "Ensure backend database queries utilize parameterized prepared statements.",
    "Verify that input sanitization filters special SQL characters like single quotes.",
    "Review authentication service audit logs for repeated injection attempts from this IP."
  ],
  "provider": "huggingface",
  "degraded": false
}
```

---

### 3.3 AI Security Log Summarization (`POST /api/v1/ai/summarize`)

#### Request Payload:
```json
{
  "logs": [
    "2026-10-29T04:00:01Z POST /login 401 Unauthorized IP=192.168.1.5",
    "2026-10-29T04:00:03Z POST /login 401 Unauthorized IP=192.168.1.5",
    "2026-10-29T04:00:07Z POST /products?id=1%20OR%201=1 200 OK IP=192.168.1.5"
  ],
  "focus": "Analyze repeated authentication failures followed by database queries"
}
```

#### Success Response (`200 OK`):
```json
{
  "summary": "Observed multiple credential stuffing attempts against /login from IP 192.168.1.5 followed by an immediate SQL Injection probe.",
  "key_findings": [
    "Repeated failed logins indicate possible brute-force attempt.",
    "Query parameter tampering detected on /products endpoint."
  ],
  "records_processed": 3,
  "provider": "huggingface",
  "degraded": false
}
```

---

### 3.4 Federated Learning Management (`POST /api/v1/federated/*`)

#### A. Start Round (`POST /api/v1/federated/rounds/start`)
- Response: Returns `round_id`, `status: "running"`, and `base_model_version`.

#### B. Submit Client Update (`POST /api/v1/federated/rounds/update`)
```json
{
  "client_id": "edge_node_london_01",
  "round_id": "8dcbb265-c357-42a6-8e11-f7b771cc4fb1",
  "base_model_version": "v1.0.0",
  "num_examples": 1500,
  "parameters": [
    [0.12, 0.45, -0.32, 0.05, 0.18, 0.91, -0.11, 0.04, 0.02, 0.15, 0.88, 1.25],
    [0.05]
  ],
  "metrics": {
    "accuracy": 0.94
  }
}
```

#### C. Run Aggregation (`POST /api/v1/federated/rounds/aggregate`)
- Computes weighted FedAvg across all accepted client updates and bumps the model version (`v1.0.0-next`).

#### D. Run Automated Multi-Client Simulation (`POST /api/v1/federated/simulate`)
```json
{
  "num_clients": 3,
  "partition_strategy": "non_iid",
  "sample_size_per_class": 50
}
```
- Returns accuracy, macro F1, and complete evaluation report for the aggregated model.

---

## 4. Standardized Error Format

All error responses from FastAPI conform to:
```json
{
  "error": {
    "code": "INVALID_FEATURES",
    "message": "Safe human-readable error description.",
    "request_id": "optional-uuid"
  }
}
```
- `400 Bad Request`: Schema validation failure or stale federated model update.
- `401 Unauthorized`: Mismatched internal API key.
- `404 Not Found`: Model or round not found.
- `500 Internal Server Error`: Generic internal error (details logged internally, never leaked over the wire).
