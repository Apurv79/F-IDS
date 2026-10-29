# F-IDS Backend Developer Guide & Integration Contract
**Target Service**: FL-IDS Python Intelligence Service (FastAPI)  
**Live Production URL**: `https://fids-ml-service.onrender.com`  
**Local Development URL**: `http://localhost:8000`  
**Interactive API Docs (Swagger UI)**: `https://fids-ml-service.onrender.com/docs`  
**Version**: `v1.0.0` (Aligned with OpenAPI 3.1.0)

---

## 1. Executive Architecture Summary

The Node.js Express backend and Python ML service communicate over strict HTTP REST contracts:

```
[ Incoming Web Traffic ]
          │
          ▼
┌──────────────────────────────────────────────┐
│         Node.js Express Backend              │
│  - Ingress middleware intercepts requests     │
│  - Tracks IP rate & failed logins via Redis  │
│  - Sanitizes request (strips auth/cookies)   │
└──────────────────────┬───────────────────────┘
                       │
       HTTP POST (Timeout 500ms, Fail-Open)
                       │
                       ▼
┌──────────────────────────────────────────────┐
│         Python FastAPI ML Service            │
│  - Decodes URLs & extracts 12 features       │
│  - Runs Logistic Regression / Random Forest  │
│  - Predicts attack class & confidence        │
└──────────────────────┬───────────────────────┘
                       │
       Prediction Response (SQLi, XSS, etc.)
                       │
                       ▼
┌──────────────────────────────────────────────┐
│         Node.js Express Backend              │
│  - If Attack: blocks/flags IP, saves alert   │
│  - Emits WebSocket alert to React frontend   │
│  - Queues BullMQ job for async AI explanation│
└──────────────────────┬───────────────────────┘
                       │
          Async BullMQ Worker Job
                       │
                       ▼
┌──────────────────────────────────────────────┐
│         Python FastAPI ML Service            │
│  - Checks Upstash Redis 24h cache            │
│  - Calls Google Gemini (gemini-3.5-flash-lite)│
│  - Returns explanation & remediation steps   │
└──────────────────────────────────────────────┘
```

---

## 2. Global Standards & Authentication

### Base URL & Environment Variables
In your Node.js `backend/.env`:
```env
ML_SERVICE_URL=https://fids-ml-service.onrender.com
ML_SERVICE_TIMEOUT_MS=500
ML_INTERNAL_API_KEY=
```

### Authentication Header
Every request from Node.js to the ML service must include:
```http
Content-Type: application/json
X-API-Key: <ML_INTERNAL_API_KEY>
X-Request-ID: <UUID-v4-Correlation-ID>
```
*(Note: If `ML_INTERNAL_API_KEY` is not set or empty in the ML service environment, key validation is automatically bypassed).*

### Correlation ID
Always generate a UUID v4 as `X-Request-ID`. The ML service will return the exact same `X-Request-ID` in the response headers and error objects for distributed log tracing.

---

## 3. Core API Contracts

---

### Endpoint 1: Health & Readiness Probe
Check if the ML service and its internal adapters are ready before processing traffic.

- **Method**: `GET`
- **Path**: `/api/v1/health/ready`
- **Headers**:
  ```http
  Accept: application/json
  ```
- **Expected Status Codes**:
  - `200 OK`: All adapters loaded and ready.
  - `503 Service Unavailable`: One or more subsystems failed initialization.

#### Response Body (`200 OK`):
```json
{
  "status": "ok",
  "service": "fl-ids-ml-service",
  "version": "0.1.0",
  "checks": {
    "prediction": "ok",
    "explanation": "ok",
    "summarization": "ok",
    "training": "ok",
    "models": "ok",
    "federated": "ok",
    "datasets": "ok",
    "experiments": "ok"
  }
}
```

---

### Endpoint 2: Real-Time Ingress Prediction (Critical Path)
Called synchronously inside the Express request middleware on every incoming HTTP request.

- **Method**: `POST`
- **Path**: `/api/v1/predictions`
- **Recommended Client Timeout**: `500ms` (Axios)
- **Failure Policy**: **Fail-Open** (if ML service times out or errors, log a warning and let the client request pass so the user experience is not disrupted).

#### Request Body (`PredictionRequest`):
> **IMPORTANT SECURITY RULE**: DO NOT send sensitive headers (e.g. `authorization`, `cookie`, `set-cookie`, `x-api-key`) in the `headers` object. Redact or strip them prior to calling the ML service.

```json
{
  "event_id": "evt_d3f9a81c",
  "node_id": "edge_client_01",
  "model_version": null,
  "request": {
    "method": "POST",
    "url": "/api/users/login?redirect=admin",
    "headers": {
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "content-type": "application/json"
    },
    "body": "{\"username\":\"admin' OR '1'='1\",\"password\":\"secret\"}",
    "source_ip": "192.168.1.105",
    "request_rate": 14.5,
    "failed_login_count": 3
  }
}
```

#### Field Specifications:
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `event_id` | `string` | Optional | Unique ID for the event |
| `node_id` | `string` | Optional | Node or Edge client identifier |
| `model_version` | `string \| null` | Optional | Specific model version (`null` uses active model) |
| `request.method` | `string` | **Yes** | HTTP Method (`GET`, `POST`, `PUT`, `DELETE`, etc.) |
| `request.url` | `string` | **Yes** | Full URL path and query parameters |
| `request.headers` | `object` | Optional | Sanitized request headers (lowercase keys) |
| `request.body` | `string \| null`| Optional | Request body payload as a string |
| `request.source_ip` | `string \| null`| Optional | Client IP address |
| `request.request_rate` | `number \| null`| Optional | Sliding window requests-per-second for this IP |
| `request.failed_login_count` | `integer \| null`| Optional | Consecutive failed logins in the last 15 min |

#### Response Body (`200 OK`):
```json
{
  "event_id": "evt_d3f9a81c",
  "prediction": "SQL Injection",
  "confidence": 0.9811,
  "model_version": "logistic_regression_v1.0.0",
  "class_probabilities": {
    "Normal": 0.0025,
    "SQL Injection": 0.9811,
    "XSS": 0.0136,
    "Brute Force": 0.0013,
    "Other/Anomaly": 0.0016
  },
  "detected_at": "2026-10-29T00:31:03.762714Z",
  "feature_contributions": [
    { "feature": "url_length", "value": 38.0 },
    { "feature": "query_parameter_count", "value": 1.0 },
    { "feature": "special_character_count", "value": 12.0 },
    { "feature": "sql_keyword_score", "value": 1.0 },
    { "feature": "payload_entropy", "value": 4.3925 }
  ]
}
```

#### Attack Classes Enum (`prediction`):
- `"Normal"`
- `"SQL Injection"`
- `"XSS"`
- `"Brute Force"`
- `"Other/Anomaly"`

#### Status Codes:
- `200 OK`: Inference succeeded.
- `422 Unprocessable Entity`: Request format failed validation.
- `503 Service Unavailable`: Model is not loaded or pipeline failed.

---

### Endpoint 3: AI Threat Explanation (Asynchronous BullMQ Path)
When a request is classified as an attack with high confidence, the Node.js backend pushes a job to a BullMQ queue to fetch an explanation without slowing down HTTP responses.

- **Method**: `POST`
- **Path**: `/api/v1/ai/explanations`
- **Recommended Client Timeout**: `15000ms` (15 seconds, due to LLM reasoning)
- **Features**: Automatically cached for 24 hours in Cloud Redis.

#### Request Body (`ExplainRequest`):
```json
{
  "event_id": "evt_d3f9a81c",
  "prediction": "SQL Injection",
  "confidence": 0.9811,
  "model_version": "logistic_regression_v1.0.0",
  "features": {
    "url_length": 38.0,
    "sql_keyword_score": 1.0,
    "payload_entropy": 4.3925
  }
}
```

#### Response Body (`200 OK`):
```json
{
  "explanation": "**Detection Summary**\nThe logistic regression model flagged this request as a SQL Injection attack (confidence 0.98). Analysis indicates deliberate manipulation of database parameters through authentication bypass syntax (' OR '1'='1).\n\n**Likely Indicators of Compromise (IoCs)**\n- Injected SQL fragment in login body\n- Elevated special character ratio\n- High keyword match on boolean query injection",
  "recommendations": [
    "Enforce parameterized queries (prepared statements) for all database queries.",
    "Validate and sanitize input using strict allow-lists.",
    "Apply WAF rules to block boolean SQL injection signatures.",
    "Review database user permissions to ensure least-privilege access."
  ],
  "provider": "gemini",
  "degraded": false
}
```

#### Status Codes:
- `200 OK`: Explanation successfully generated (or served from Redis cache).
- `422 Unprocessable Entity`: Missing required fields (`prediction`, `confidence`, `model_version`).
- `500 Internal Server Error`: AI providers failed and no cache was found.

---

### Endpoint 4: Security Log Summarization (Admin Dashboard Batch)
Used by the Admin portal to generate an executive intelligence summary of the last $N$ security logs.

- **Method**: `POST`
- **Path**: `/api/v1/ai/summaries`
- **Recommended Client Timeout**: `30000ms` (30 seconds)

#### Request Body (`SummarizeRequest`):
```json
{
  "logs": [
    "2026-10-29T05:10:00Z IP=192.168.1.100 METHOD=POST URL=/login STATUS=401 ATTEMPT=1",
    "2026-10-29T05:10:02Z IP=192.168.1.100 METHOD=POST URL=/login STATUS=401 ATTEMPT=2",
    "2026-10-29T05:10:04Z IP=192.168.1.100 METHOD=POST URL=/login STATUS=401 ATTEMPT=3",
    "2026-10-29T05:12:15Z IP=10.0.0.5 METHOD=GET URL=/products?cat=1%27%20UNION%20SELECT%20NULL-- STATUS=200"
  ],
  "focus": "Credential stuffing and SQL injection activity in the past hour"
}
```

#### Response Body (`200 OK`):
```json
{
  "summary": "Observed targeted credential brute-forcing against /login from 192.168.1.100 alongside database reconnaissance from 10.0.0.5.",
  "key_findings": [
    "Rapid consecutive failed logins indicative of automated brute force attack.",
    "Union-based SQL injection targeting product catalog."
  ],
  "records_processed": 4,
  "provider": "gemini",
  "degraded": false
}
```

---

### Endpoint 5: Federated Learning Round Management

#### 5.1 Create a Round:
- **Method**: `POST`
- **Path**: `/api/v1/federated/rounds`
- **Body**:
  ```json
  {
    "round_number": 1,
    "num_clients": 3,
    "strategy": "fedavg",
    "min_client_fraction": 1.0
  }
  ```
- **Response (`201 Created`)**:
  ```json
  {
    "round_id": "round_001",
    "status": "running",
    "base_model_version": "v1.0.0",
    "participating_clients": 3,
    "completed_clients": 0
  }
  ```

#### 5.2 Submit Client Local Updates:
- **Method**: `POST`
- **Path**: `/api/v1/federated/rounds/{round_id}/updates`
- **Body**:
  ```json
  {
    "round_id": "round_001",
    "client_id": "client_alpha",
    "base_model_version": "v1.0.0",
    "parameters": {
      "coef": [[0.12, -0.45, 0.88, 0.05, 0.32]],
      "intercept": [0.01]
    },
    "num_examples": 250,
    "metrics": {
      "accuracy": 0.94,
      "loss": 0.12
    }
  }
  ```
- **Response (`202 Accepted`)**:
  ```json
  {
    "client_id": "client_alpha",
    "round_id": "round_001",
    "accepted": true
  }
  ```

#### 5.3 Trigger FedAvg Aggregation:
- **Method**: `POST`
- **Path**: `/api/v1/federated/rounds/{round_id}/aggregate`
- **Body**:
  ```json
  {
    "force": false
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "round_id": "round_001",
    "status": "completed",
    "base_model_version": "v1.0.0",
    "resulting_model_version": "v1.1.0",
    "participating_clients": 3,
    "completed_clients": 3
  }
  ```

---

## 4. Backend Implementation Plan (Step-by-Step for Node.js)

Here is the exact implementation structure for the Node.js developer:

### Phase 1: Implement `src/services/ml.service.ts`
Implement the Axios communication wrapper with retries, timeouts, and error handling.

```typescript
// backend/src/services/ml.service.ts
import axios, { AxiosInstance } from "axios";
import { v4 as uuidv4 } from "uuid";

export interface IngressContext {
  method: string;
  url: string;
  headers?: Record<string, string>;
  body?: string;
  sourceIp?: string;
  requestRate?: number;
  failedLoginCount?: number;
}

export interface PredictionResult {
  eventId: string;
  prediction: "Normal" | "SQL Injection" | "XSS" | "Brute Force" | "Other/Anomaly";
  confidence: number;
  modelVersion: string;
  classProbabilities: Record<string, number>;
  featureContributions: Array<{ feature: string; value: number }>;
}

export interface ExplainResult {
  explanation: string;
  recommendations: string[];
  provider: string;
  degraded: boolean;
}

class MLService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.ML_SERVICE_URL || "https://fids-ml-service.onrender.com",
      timeout: Number(process.env.ML_SERVICE_TIMEOUT_MS) || 500, // 500ms for inference
      headers: {
        "Content-Type": "application/json",
        ...(process.env.ML_INTERNAL_API_KEY && {
          "X-API-Key": process.env.ML_INTERNAL_API_KEY,
        }),
      },
    });
  }

  /**
   * Synchronous real-time attack prediction
   * Implements FAIL-OPEN policy so legitimate traffic is never blocked if ML is down.
   */
  async predict(context: IngressContext): Promise<PredictionResult | null> {
    const requestId = uuidv4();
    try {
      // 1. Sanitize headers to prevent sending tokens or cookies
      const sanitizedHeaders = { ...(context.headers || {}) };
      delete sanitizedHeaders["authorization"];
      delete sanitizedHeaders["cookie"];
      delete sanitizedHeaders["x-api-key"];

      const payload = {
        event_id: `evt_${requestId.substring(0, 8)}`,
        request: {
          method: context.method,
          url: context.url,
          headers: sanitizedHeaders,
          body: typeof context.body === "string" ? context.body : JSON.stringify(context.body || ""),
          source_ip: context.sourceIp,
          request_rate: context.requestRate || 0,
          failed_login_count: context.failedLoginCount || 0,
        },
      };

      const res = await this.client.post("/api/v1/predictions", payload, {
        headers: { "X-Request-ID": requestId },
      });

      return {
        eventId: res.data.event_id,
        prediction: res.data.prediction,
        confidence: res.data.confidence,
        modelVersion: res.data.model_version,
        classProbabilities: res.data.class_probabilities,
        featureContributions: res.data.feature_contributions,
      };
    } catch (error: any) {
      console.warn(`[ML-Service] Inference failed or timed out (${error.message}). Failing open.`);
      return null;
    }
  }

  /**
   * Asynchronous AI Threat Explanation (called via BullMQ worker)
   */
  async explainAttack(params: {
    prediction: string;
    confidence: number;
    modelVersion: string;
    features?: Record<string, any>;
  }): Promise<ExplainResult | null> {
    try {
      const res = await this.client.post(
        "/api/v1/ai/explanations",
        {
          prediction: params.prediction,
          confidence: params.confidence,
          model_version: params.modelVersion,
          features: params.features || {},
        },
        {
          timeout: 15000, // 15 seconds for LLM reasoning
          headers: { "X-Request-ID": uuidv4() },
        }
      );
      return res.data;
    } catch (error: any) {
      console.error(`[ML-Service] AI Explanation error:`, error.message);
      return null;
    }
  }
}

export const mlService = new MLService();
```

---

### Phase 2: Ingress Inspection Middleware
Mount `ml.middleware.ts` before your route controllers in `src/app.ts`.

```typescript
// backend/src/middlewares/ml.middleware.ts
import { Request, Response, NextFunction } from "express";
import { mlService } from "../services/ml.service";
import { redisService } from "../services/redis.service"; // for rate/login telemetry

export async function idsInspectionMiddleware(req: Request, res: Response, next: NextFunction) {
  // Skip static assets or internal healthchecks
  if (req.path.startsWith("/health") || req.path.startsWith("/public")) {
    return next();
  }

  const clientIp = req.ip || req.socket.remoteAddress || "127.0.0.1";

  // Fetch telemetry from Redis
  const requestRate = await redisService.getRequestRate(clientIp);
  const failedLogins = await redisService.getFailedLoginCount(clientIp);

  // Call ML Service (guaranteed 500ms timeout)
  const result = await mlService.predict({
    method: req.method,
    url: req.originalUrl,
    headers: req.headers as Record<string, string>,
    body: req.body ? JSON.stringify(req.body) : undefined,
    sourceIp: clientIp,
    requestRate,
    failedLoginCount: failedLogins,
  });

  if (result && result.prediction !== "Normal" && result.confidence >= 0.85) {
    console.warn(`[F-IDS ALERT] ${result.prediction} detected from ${clientIp} (conf: ${result.confidence})`);
    
    // 1. Dispatch async BullMQ job to get Gemini explanation & store in DB
    // await queueService.add("explain-attack", { ...result, sourceIp: clientIp, url: req.originalUrl });

    // 2. Action: In strict mode, block attack. In monitor mode, attach header and allow through.
    if (process.env.IDS_BLOCK_MODE === "true") {
      return res.status(403).json({
        error: "ACCESS_DENIED",
        message: "Request blocked by Federated Intrusion Detection System.",
        incident_id: result.eventId,
      });
    }
  }

  next();
}
```

---

## 5. Error Schema Reference
When an error occurs, the ML service returns this standard structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR | ROUND_NOT_FOUND | UNAUTHORIZED | INTERNAL_ERROR",
    "message": "Human readable error description",
    "details": [ ... ],
    "request_id": "c62b487e-e21e-450f-90db-2b509bc923a1"
  }
}
```

---

## 6. Quick Checklist for the Backend Developer

- [ ] Add `ML_SERVICE_URL=https://fids-ml-service.onrender.com` to `backend/.env`.
- [ ] Implement `src/services/ml.service.ts` using the provided code template.
- [ ] Ensure request headers strip `authorization` and `cookie` before calling `/api/v1/predictions`.
- [ ] Configure Axios timeout for predictions to `500ms` with fail-open error catching.
- [ ] Set up BullMQ queue for `/api/v1/ai/explanations` with a `15s` timeout.
- [ ] Test the pipeline by hitting your Express backend with a simulated SQLi (`GET /login?user=admin' OR 1=1--`) and verify the alert event.
