"""Integration points for model-based explainability.

SHAP/LIME explanations are different from AI-generated natural-language
attack explanations, which belong in the ai/ module.
"""


def explain_with_shap(*args, **kwargs):
    """Implement SHAP-based feature attribution in a later phase."""
    raise NotImplementedError(
        "SHAP integration is planned for a later phase"
    )


def explain_with_lime(*args, **kwargs):
    """Implement LIME-based feature attribution in a later phase."""
    raise NotImplementedError(
        "LIME integration is planned for a later phase"
    )
