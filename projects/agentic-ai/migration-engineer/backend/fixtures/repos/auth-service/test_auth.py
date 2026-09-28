"""Behavioural check for auth-service. Exit 0 = pass, non-zero = fail."""

import sys
from pathlib import Path

from auth import token_expiry, token_issued_at


def main() -> None:
    issued = token_issued_at()
    assert issued.tzinfo is not None, "token_issued_at() must be timezone-aware"
    expiry = token_expiry(2)
    assert expiry.tzinfo is not None, "token_expiry() must be timezone-aware"
    assert expiry > issued, "expiry must be after issue time"
    src = Path(__file__).with_name("auth.py").read_text(encoding="utf-8")
    assert "datetime.utcnow" not in src, "deprecated datetime.utcnow() is still present in auth.py"
    print("PASS: auth-service timestamps are timezone-aware")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        sys.exit(1)
