"""Behavioural check for metrics-service. Exit 0 = pass, non-zero = fail.

Note: failure messages deliberately avoid the literal deprecated call so a learner's
detect regex only matches the real source, keeping the exercise diff clean.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from metrics import event_from_epoch, window_start


def main() -> None:
    epoch = event_from_epoch(0)
    assert epoch.tzinfo is not None, "event_from_epoch() must return a timezone-aware datetime"
    assert epoch == datetime(1970, 1, 1, tzinfo=timezone.utc), "epoch 0 must map to 1970-01-01 UTC"
    ws = window_start(125, 60)
    assert ws.tzinfo is not None, "window_start() must return a timezone-aware datetime"
    assert ws.second == 0 and ws.minute == 1, "window_start(125, 60) must floor to the minute"
    src = Path(__file__).with_name("metrics.py").read_text(encoding="utf-8")
    assert "utcfromtimestamp" not in src, "the deprecated naive-UTC epoch call is still present in metrics.py"
    print("PASS: metrics-service timestamps are timezone-aware")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        sys.exit(1)
