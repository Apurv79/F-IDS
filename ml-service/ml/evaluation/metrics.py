"""Classification metrics for FL-IDS model evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.dataset.labels import CANONICAL_LABELS


def evaluate_classifier(
    model,
    X_test,
    y_test,
) -> dict:
    """Evaluate a fitted classifier on held-out test data.

    Returns overall metrics, per-class metrics, a confusion matrix,
    and ROC-AUC where the available data supports its calculation.
    """
    predictions = model.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "per_class": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    # Calculate ROC-AUC if probability prediction is available
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)
            classes = getattr(model, "classes_", np.unique(y_test))
            if len(classes) == probabilities.shape[1] and len(classes) > 1:
                result["roc_auc_ovr"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except Exception:
            result["roc_auc_ovr"] = None
    else:
        result["roc_auc_ovr"] = None

    return result
