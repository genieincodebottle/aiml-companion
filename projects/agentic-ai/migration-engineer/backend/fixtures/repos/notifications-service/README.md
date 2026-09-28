# notifications-service (fixture)

Stand-in service for the migration demo. Uses deprecated `datetime.utcnow()` (including
in an `.isoformat()` chain) without importing `timezone`, so the migration takes two
passes. Check: `python test_notify.py`.
