# Mission Control - backend

The platform: AI gateway, policy engine, workload identity, prompt and agent
registries, the durable runtime with the reviewer gate, the evaluation plane and
tracing, plus the reference **fincrime** tenant. Runs offline with no key.

```bash
uv sync --extra dev
uv run uvicorn main:app --reload --port 8000   # the API
uv run python cli.py run ALRT-001              # one case through the gate
uv run python -m evaluation.run_eval           # every golden case + the promotion gate
uv run pytest -q                               # offline; never calls a model
```

Real OpenTelemetry export needs `uv sync --extra dev --extra otel` and
`MC_OTEL_EXPORTER=console` (or `otlp`). See the top-level [README](../README.md) and
[docs/](../docs/).
