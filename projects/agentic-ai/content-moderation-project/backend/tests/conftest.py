"""pytest collection settings.

test_e2e.py is a standalone script that drives a live server on localhost:8000
(`python tests/test_e2e.py`). Under pytest its helper `test(...)` is collected as
a test and errors, and its scenarios time out against a server that is not
running, so pytest skips it.
"""

collect_ignore = ["test_e2e.py"]
