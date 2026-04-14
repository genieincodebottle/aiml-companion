"""Test configuration. Tests default to US country profile for consistent PII/HITL behavior."""

import os

# Set default test country to US so PII masking tests (SSN, US phone patterns)
# and HITL threshold tests ($10K) pass regardless of what .env says.
os.environ["COUNTRY"] = "us"
