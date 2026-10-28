# FL-IDS ML Module

The ML module detects web attacks using supervised machine learning.

## Classification classes

- Normal
- SQL Injection
- XSS
- Brute Force
- Other/Anomaly

## Module structure

- `ml/dataset/`: dataset loading, validation, label normalization,
  synthetic examples, and dataset versioning.
- `ml/features/`: HTTP feature extraction and preprocessing.
- `ml/models/`: classifier creation, pipelines, model artifacts,
  and model metadata.
- `ml/training/`: reproducible training and experiment tracking.
- `ml/evaluation/`: metrics, reports, and explainability hooks.
- `tests/ml/`: automated tests for ML components.

## Supported algorithms

1. Logistic Regression — baseline.
2. Random Forest — comparison model.
3. MLP — optional later experiment.

## Evaluation

The pipeline reports accuracy, macro precision, macro recall,
macro F1, weighted F1, a confusion matrix, and per-class metrics.
ROC-AUC is calculated when supported by the test data.

## Run tests

From the `ml-service` directory:

    uv run pytest tests/ml -v

## Train a model

After preparing a CSV with the expected request columns and a
supported label column:

    uv run python -m ml.training.train \
        --csv path/to/dataset.csv \
        --label-column label

To select another algorithm:

    uv run python -m ml.training.train \
        --csv path/to/dataset.csv \
        --algorithm random_forest

## Important limitations

Synthetic examples are intended for development and smoke testing,
not production accuracy claims.

Public datasets require dataset-specific adapters before use.
Federated aggregation belongs in the `federated/` module.
AI-generated explanations belong in the `ai/` module.
SHAP/LIME support is planned for a later phase.
