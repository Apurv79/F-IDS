"""Inference and attack prediction routes."""

import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
import pandas as pd

from api.dependencies.auth import verify_internal_api_key
from ml.dataset.synthetic import generate_http_samples
from ml.features.extractor import FEATURE_COLUMNS, extract_features
from ml.models.pipeline import build_training_pipeline
from shared.exceptions import InvalidFeaturesError
from shared.schemas.prediction import PredictionRequest, PredictionResponse
from shared.utils.request_id import generate_request_id

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
    dependencies=[Depends(verify_internal_api_key)],
)

# In-memory default active model pipeline (trained on synthetic baseline at launch)
_ACTIVE_PIPELINE = None
_ACTIVE_MODEL_VERSION = "logistic_regression_v1.0.0"


def get_or_init_pipeline():
    global _ACTIVE_PIPELINE
    if _ACTIVE_PIPELINE is None:
        dataset = generate_http_samples(rows_per_class=50, seed=42)
        records = dataset.to_dict(orient="records")
        features_list = [extract_features(r) for r in records]
        X = pd.DataFrame(features_list, columns=FEATURE_COLUMNS).fillna(0.0)
        y = dataset["label"]

        pipeline = build_training_pipeline("logistic_regression", random_seed=42)
        pipeline.fit(X, y)
        _ACTIVE_PIPELINE = pipeline
    return _ACTIVE_PIPELINE


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Synchronous Real-Time Attack Classification",
)
async def predict_attack(request: PredictionRequest) -> PredictionResponse:
    """Predict whether an incoming HTTP request is Normal or an attack.

    Accepts raw HTTP data directly from Node.js Express or pre-extracted features.
    """
    req_id = generate_request_id()

    # 1. Resolve features
    if request.raw_request is not None:
        raw_dict = {
            "url": request.raw_request.url,
            "method": request.raw_request.method,
            "headers": request.raw_request.headers,
            "body": (
                json.dumps(request.raw_request.body)
                if isinstance(request.raw_request.body, dict)
                else str(request.raw_request.body)
            ),
            "ip_request_rate": request.raw_request.ip_request_rate,
            "failed_login_count": request.raw_request.failed_login_count,
        }
        extracted = extract_features(raw_dict)
    elif request.features:
        extracted = {col: float(request.features.get(col, 0.0)) for col in FEATURE_COLUMNS}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_INPUT",
                "message": "Either 'raw_request' or 'features' must be provided.",
            },
        )

    # 2. Convert to model-ready DataFrame
    X_single = pd.DataFrame([extracted], columns=FEATURE_COLUMNS).fillna(0.0)

    pipeline = get_or_init_pipeline()
    prediction_label = pipeline.predict(X_single)[0]

    # Calculate probabilities and confidence
    class_probs: dict[str, float] = {}
    confidence = 1.0
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_single)[0]
        classes = pipeline.classes_
        for cls_name, prob in zip(classes, probabilities, strict=False):
            class_probs[str(cls_name)] = round(float(prob), 4)
        confidence = round(float(np.max(probabilities)), 4)

    return PredictionResponse(
        prediction=prediction_label,
        confidence=confidence,
        model_version=_ACTIVE_MODEL_VERSION,
        class_probabilities=class_probs,
        event_id=request.event_id,
        request_id=req_id,
    )
