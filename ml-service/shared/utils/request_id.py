"""Generate correlation IDs for tracing requests and events."""

from uuid import uuid4


def generate_request_id() -> str:
    """Generate a unique identifier for one request.

    This ID helps correlate API logs, model predictions,
    AI explanations, and errors. It is not an authentication token.
    """
    return str(uuid4())
