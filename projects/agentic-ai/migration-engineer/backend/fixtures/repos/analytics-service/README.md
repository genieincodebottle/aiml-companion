# analytics-service (fixture)

Stand-in service for the migration demo. Two modules (`events.py`, `sessions.py`) both
use deprecated `datetime.utcnow()` without importing `timezone`, so the worker edits
multiple files and iterates until the shared check passes. Check: `python test_analytics.py`.
