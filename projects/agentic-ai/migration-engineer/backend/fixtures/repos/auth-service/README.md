# auth-service (fixture)

Stand-in service for the migration demo. Uses deprecated `datetime.utcnow()` and does
NOT import `timezone`, so the correct migration takes two passes (edit, fail, add
import, pass) - which is what makes the agent loop visibly iterate. Check:
`python test_auth.py`.
