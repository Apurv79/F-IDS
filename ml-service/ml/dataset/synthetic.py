"""Generate small, reproducible HTTP request examples.

These examples are for testing code and are not a substitute for
validated cybersecurity datasets or reliable performance evaluation.
"""

import random

import pandas as pd

_NORMAL_PATHS = [
    "/",
    "/home",
    "/products?id=12",
    "/search?q=shoes",
    "/login",
]

_SQL_PATHS = [
    "/products?id=1%20OR%201=1",
    "/search?q=%27%20UNION%20SELECT%20NULL",
]

_XSS_PATHS = [
    "/search?q=%3Cscript%3Ealert(1)%3C/script%3E",
    "/profile?name=%3Cimg%20src=x%20onerror=alert(1)%3E",
]

_BRUTE_PATHS = [
    "/login",
    "/auth/signin",
]

_ANOMALY_PATHS = [
    "/api/items",
    "/api/search",
]


def generate_http_samples(
    rows_per_class: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate labeled HTTP examples using a reproducible random seed."""
    if rows_per_class < 1:
        raise ValueError("rows_per_class must be at least 1")

    rng = random.Random(seed)
    templates = [
        ("Normal", _NORMAL_PATHS),
        ("SQL Injection", _SQL_PATHS),
        ("XSS", _XSS_PATHS),
        ("Brute Force", _BRUTE_PATHS),
        ("Other/Anomaly", _ANOMALY_PATHS),
    ]

    records = []
    for label, paths in templates:
        for _ in range(rows_per_class):
            is_brute_force = label == "Brute Force"
            is_anomaly = label == "Other/Anomaly"
            method = "POST" if is_brute_force else rng.choice(["GET", "POST", "PUT"])
            url = rng.choice(paths)
            body = '{"username": "admin", "password": "123"}' if is_brute_force else ""
            ip_request_rate = rng.randint(50, 200) if (is_brute_force or is_anomaly) else rng.randint(1, 10)
            failed_login_count = rng.randint(5, 30) if is_brute_force else 0

            records.append({
                "url": url,
                "method": method,
                "body": body,
                "ip_request_rate": ip_request_rate,
                "failed_login_count": failed_login_count,
                "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                "label": label,
            })

    return pd.DataFrame(records)
