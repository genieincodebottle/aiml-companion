"""Behavioural check for analytics-service (two modules must both be migrated)."""

import sys
from pathlib import Path

from events import event_time
from sessions import session_start


def main() -> None:
    assert event_time().tzinfo is not None, "event_time() must be timezone-aware"
    assert session_start().tzinfo is not None, "session_start() must be timezone-aware"
    here = Path(__file__).parent
    for mod in ("events.py", "sessions.py"):
        src = (here / mod).read_text(encoding="utf-8")
        assert "datetime.utcnow" not in src, f"deprecated datetime.utcnow() is still present in {mod}"
    print("PASS: analytics-service timestamps are timezone-aware")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        sys.exit(1)
