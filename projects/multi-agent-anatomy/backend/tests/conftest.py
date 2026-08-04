"""Force replay mode for the whole test suite.

A key in .env must not turn the tests into live API calls. They would be
nondeterministic, they would cost money on every run, and the assertions here
are about the pipeline's behaviour rather than the model's prose.

This has to run before app.config is imported, which is why it lives in
conftest.py at module scope rather than in a fixture.
"""

import os

os.environ["REPLAY_ONLY"] = "true"
