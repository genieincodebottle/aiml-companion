"""Billing timestamp helpers for the billing-service.

NOTE (for the demo): this module intentionally uses the deprecated naive-UTC call.
`timezone` is already imported here, so a correct migration is a SINGLE-pass fix.
"""

from datetime import datetime, timezone  # noqa: F401 - timezone used after migration


def invoice_issued_at() -> datetime:
    # The naive-UTC call below is deprecated and removed in a future Python.
    return datetime.utcnow()


def payment_captured_at() -> datetime:
    return datetime.utcnow()
