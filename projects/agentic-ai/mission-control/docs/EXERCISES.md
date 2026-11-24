# Teaching labs

Each lab isolates one production concern. All run offline with no key. Run them from
`backend/` after `uv sync --extra dev`.

> The `VAR=1 command` prefix is bash syntax. In Windows PowerShell set the variable
> first: `$env:MC_DISABLE_POLICY="1"; uv run python cli.py run ALRT-001`, then
> `Remove-Item Env:MC_DISABLE_POLICY` afterwards.

## Lab 1 - The governance contrast (policy engine)

Run a normal case and note the reviewer gate:

```bash
uv run python cli.py run ALRT-001
# -> [GATE] ... is waiting for a reviewer: case.file_sar
```

Now switch the policy engine off and run the same agents again. The risky
`case.file_sar` disposition commits with nobody in the loop:

```bash
MC_DISABLE_POLICY=1 uv run python cli.py run ALRT-001
# -> status: resolved (no gate, no reviewer)
```

**The lesson:** governance is a property of the platform, not of the agent. Turn the
platform's policy plane off and the same agents become unsafe.

## Lab 2 - A case that survives a restart

Open a case and leave it parked at the gate. The CLI decides at once, so use the API:

```bash
uv run uvicorn main:app --port 8000          # terminal 1
curl -s -X POST localhost:8000/api/investigations -H "X-Operator: ana" \
     -H "Content-Type: application/json" -d '{"alert_id":"ALRT-002"}'
```

Stop the server (Ctrl+C), start it again, and decide the case:

```bash
uv run python cli.py cases                                  # it is still there
uv run python cli.py decide <workflow-id> approve --as rex  # and it completes
```

Compare the `gateway_call` events before and after in
`.work/events/<workflow-id>.jsonl`: nothing re-ran, nothing was billed twice. The
runtime rebuilt the case from `.work/checkpoints/<workflow-id>.json` (see `_rehydrate`
in [`runtime/engine.py`](../backend/src/runtime/engine.py)).

## Lab 3 - Saga compensation

```bash
uv run python cli.py run ALRT-001 --fault
# -> saga: commit failed -> compensated -> escalated to a person
```

The notify step failed after the case record was written, so the saga removed the
record again. No half-filed SAR is left behind. Compare the event logs of a clean run
and a `--fault` run: only the fault run has `saga_compensated`.

## Lab 4 - The promotion gate moves a real label

```bash
uv run python cli.py prompts                        # every prompt at production=v1
uv run python -m evaluation.run_eval --candidate evaluation/candidates/narrative_judgement.json
uv run python cli.py prompts                        # candidate=v2, production still v1
```

Offline, the gate refuses to promote: the offline brain ignores wording, so the run
cannot have measured the new prompt. With a key set, the same command runs every case
live. This candidate drops the desk's written disposition policy, so it usually gets
the PEP case wrong and the gate fails. Write a better candidate and watch production
move; `cli.py prompts rollback fincrime.narrative 1` moves it back.

## Lab 5 - Semantic cache

The gateway serves near-duplicate prompts from cache for free. See
[`tests/test_gateway.py`](../backend/tests/test_gateway.py) and vary a prompt until
cosine similarity drops below `MC_CACHE_SIMILARITY` and the cache misses. Then find two
prompts with opposite meanings ("transfer approved" / "transfer denied") that the
cache would serve for each other. What did that cost?

## Lab 6 - Beat the injection scanner

ALRT-004's customer note (entity `E400` in
[`tenants/fincrime/data.py`](../backend/src/tenants/fincrime/data.py)) tells the agent to
close the case. The scanner catches it and the case is still filed. Rewrite the note in
words the pattern list misses, run the case live, and see whether the agent obeys it.
Then decide: add a pattern, or argue the architectural controls (closed tool lists, the
reviewer gate) make it not matter?

## Lab 7 - Maker-checker with real logins

```bash
MC_OPERATORS="ana:analyst:t-ana,rex:reviewer:t-rex,boss:analyst+reviewer:t-boss" \
  uv run uvicorn main:app --port 8000
```

Open a case with `Authorization: Bearer t-boss`, then try to approve it with the same
token. `boss` holds both roles and is still refused (403): maker-checker is about the
person, not the role. Approve it as `rex` and read `decided_by` on the case.

## Lab 8 - Add a second tenant

Implement a new `Workload` ([`runtime/workflow.py`](../backend/src/runtime/workflow.py)),
even a two-step one, register its agent cards, and add it to `WORKLOADS` in
[`control_plane.py`](../backend/src/control_plane.py). Nothing in the gateway, policy,
runtime or eval plane changes. That invariance is the platform.
