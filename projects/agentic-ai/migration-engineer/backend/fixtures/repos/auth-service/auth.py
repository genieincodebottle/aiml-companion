"""Auth token timing for the auth-service.

NOTE (for the demo): uses the deprecated naive-UTC call and does NOT import
`timezone`. A correct migration therefore needs TWO passes: replace the calls, watch
the tests fail with NameError, then add the import.
"""

from datetime import datetime, timedelta


def token_issued_at() -> datetime:
    return datetime.utcnow()


def token_expiry(hours: int = 1) -> datetime:
    return datetime.utcnow() + timedelta(hours=hours)
