"""Extract numeric machine-learning features from HTTP request records."""

from collections import Counter
import math
import re
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

FEATURE_COLUMNS = [
    "url_length",
    "query_parameter_count",
    "special_character_count",
    "sql_keyword_score",
    "payload_entropy",
    "http_method_code",
    "ip_request_rate",
    "failed_login_count",
    "ua_anomaly",
    "body_length",
    "has_script_tag",
    "has_sql_operator_pattern",
]

METHOD_CODES = {
    "GET": 1,
    "POST": 2,
    "PUT": 3,
    "DELETE": 4,
    "PATCH": 5,
    "HEAD": 6,
    "OPTIONS": 7,
}

_SQL_KEYWORDS = [
    "select",
    "insert",
    "update",
    "delete",
    "drop",
    "union",
    "where",
    "or",
    "and",
    "from",
    "table",
    "information_schema",
    "exec",
    "concat",
]


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy for a given string."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def extract_features(request: dict[str, Any]) -> dict[str, float]:
    """Extract numeric features from an HTTP request dict."""
    url = str(request.get("url") or "")
    method = str(request.get("method") or "GET").upper()
    body = str(request.get("body") or "")
    headers = request.get("headers") or {}

    # Parse URL
    parsed = urlparse(url)
    query = parsed.query
    query_count = len(query.split("&")) if query else 0

    # Special characters in URL and body
    decoded_url = unquote(url)
    decoded_body = unquote(body)
    combined = f"{url} {body} {decoded_url} {decoded_body}"
    special_chars = len(re.findall(r"['\";<>()=#*|&%\$\\]", combined))

    # SQL keyword score
    lower_content = combined.lower()
    keyword_score = sum(
        1 for kw in _SQL_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", lower_content)
    )

    # User-agent anomaly
    user_agent = ""
    if isinstance(headers, dict):
        for k, v in headers.items():
            if k.lower() == "user-agent":
                user_agent = str(v)
                break
    ua_anomaly = 1 if (not user_agent or len(user_agent) < 10) else 0

    # Script tag detection
    script_tag = int(
        bool(
            re.search(
                r"<\s*script\b|javascript\s*:",
                combined,
                re.IGNORECASE,
            )
        )
    )

    # Detect common SQL operator patterns
    sql_pattern = int(
        bool(
            re.search(
                r"\b(or|and)\b\s+['\"\d]+\s*=\s*['\"\d]+"
                r"|union\s+select",
                combined,
                re.IGNORECASE,
            )
        )
    )

    return {
        "url_length": float(len(url)),
        "query_parameter_count": float(query_count),
        "special_character_count": float(special_chars),
        "sql_keyword_score": float(keyword_score),
        "payload_entropy": float(shannon_entropy(body or url)),
        "http_method_code": float(METHOD_CODES.get(method, 7)),
        "ip_request_rate": float(request.get("ip_request_rate", 0) or 0),
        "failed_login_count": float(request.get("failed_login_count", 0) or 0),
        "ua_anomaly": float(ua_anomaly),
        "body_length": float(len(body)),
        "has_script_tag": float(script_tag),
        "has_sql_operator_pattern": float(sql_pattern),
    }


def extract_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of request records into numeric features."""
    records = frame.to_dict(orient="records")
    features = [extract_features(record) for record in records]
    # Reindex explicitly to preserve the model's expected column order.
    return pd.DataFrame(
        features,
        columns=FEATURE_COLUMNS,
    ).fillna(0.0)
