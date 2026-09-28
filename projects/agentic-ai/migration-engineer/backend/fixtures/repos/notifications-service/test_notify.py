"""Behavioural check for notifications-service. Exit 0 = pass, non-zero = fail."""

import sys
from pathlib import Path

from notify import queued_at, queued_at_iso


def main() -> None:
    assert queued_at().tzinfo is not None, "queued_at() must be timezone-aware"
    iso = queued_at_iso()
    assert "+00:00" in iso, "queued_at_iso() must carry a UTC offset once migrated"
    src = Path(__file__).with_name("notify.py").read_text(encoding="utf-8")
    assert "datetime.utcnow" not in src, "deprecated datetime.utcnow() is still present in notify.py"
    print("PASS: notifications-service timestamps are timezone-aware")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        sys.exit(1)
