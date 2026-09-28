# billing-service (fixture)

A tiny stand-in service used by the Autonomous Migration Engineer demo. It contains
the deprecated `datetime.utcnow()` the agent is asked to modernize. `timezone` is
already imported, so the correct migration is a single edit. Run its check with
`python test_billing.py`.
