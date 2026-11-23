# Demo guide

Five demos, ordered by what they cost you. All of them run without an API key
(the console uses the labelled offline agent); a key makes demos 4 and 5 show real
Gemini reasoning for about a cent per incident.

| # | Demo | Needs a key | Time | Cost |
|---|---|---|---|---|
| 1 | [The loop, offline](#1-the-loop-offline) | no | ~3s | free |
| 2 | [The guardrails, offline](#2-the-guardrails-offline) | no | ~2 min | free |
| 3 | [The notebook](#3-the-notebook) | no | ~15 min | free |
| 4 | [The live console](#4-the-live-console) | optional | ~2 min | free offline, ~$0.012 live |
| 5 | [Persistence + login](#5-persistence-and-login) | optional, + Docker | ~3 min | free offline, ~$0.012 live |

If you are showing this to someone else, demo 4 is the one that lands, and demo 1 is
what you run first so they know it is not a puppet show. Demo 5 is optional and
exists to show that persistence and login are *additions*, not requirements.

---

## Setup

```bash
cd projects/agentic-ai/incident-commander/backend
uv sync --extra dev
```

That is all any demo needs. For real Gemini reasoning in demos 4 and 5, add a key:

```bash
cp .env.example .env     # then set GOOGLE_API_KEY in it (Windows: copy .env.example .env)
```

Free key: <https://aistudio.google.com/apikey>.

---

## 1. The loop, offline

Run all ten fault scenarios end to end with a deterministic stub in place of
Gemini:

```bash
uv run python -m evaluation.run_eval --stub
```

You get the full evaluation report: outcome rate, correct-first-action, iteration
counts, harmful-action rate, injection-resisted rate, trajectory scores.

**What to point at.** The last four scenarios in the table. `unfixable_ca_expiry`
and `false_alarm` are scored on the agent *not* acting - an agent measured only on
faults it can fix will cheerfully invent a fix for one it cannot, and no outcome
metric will ever catch it. `silent_config_drift` is unsolvable from the default
telemetry bundle; the agent has to decide it needs a tool it was not handed.

**What this does not show.** The stub measures the harness, not the model. The
report says so in its own header. That is the honest framing to use.

---

## 2. The guardrails, offline

The interesting demo is not that it works. It is what happens when you take
something away.

Baseline for all of these is **100% correct outcome, trajectory 1.00**.

```bash
# Strip the agent's permissions. RBAC refuses every remediation.
IC_AGENT_ROLE=read-only uv run python -m evaluation.run_eval --stub
#   -> 20% outcome, trajectory 0.77. The 20% is the two negative scenarios,
#      which pass by not acting and so are unaffected.

# Nonsense role. Fail closed: no permissions at all, not a default set.
IC_AGENT_ROLE=captain uv run python -m evaluation.run_eval --stub
#   -> 10% outcome, 0% correct-first-action. Strictly worse than read-only,
#      which is the point: an unknown role is more dangerous than a known
#      restricted one, so it gets less.

# Cut the evidence budget to zero.
IC_MAX_EVIDENCE_ROUNDS=0 uv run python -m evaluation.run_eval --stub
#   -> 90%. silent_config_drift escalates, because its cause is not in the
#      default telemetry bundle and the agent can no longer go and look.

# One iteration only.
IC_MAX_ITERATIONS=1 uv run python -m evaluation.run_eval --stub
#   -> 90%. cascading_failure needs two ordered fixes and now hits the ceiling
#      mid-repair. Correct-first-action stays at 100%: it was not wrong, it
#      was cut off.
```

Predicting the number before you run it is the actual exercise.
[EXERCISES.md](EXERCISES.md) has 17 of these with the reasoning written out.

Two switches deliberately do **not** move the headline metrics, and that is worth
demoing too:

```bash
IC_QUARANTINE_INJECTIONS=false uv run python -m evaluation.run_eval --stub \
  --scenarios poisoned_telemetry --out /tmp/flagonly.json
```

Still 100%. The change is in `guardrail_flags` inside the JSON: `"the imperative
content was removed and the remainder fenced as untrusted data"` becomes
`"detected ... (flag-only mode)"`. The agent got the right answer either way -
against a stub that was never going to obey the injection. That gap between
*detecting* and *defending* is exactly what a pass rate cannot show you
([ADR-0007](docs/adr/0007-quarantine-not-detection.md)).

```bash
IC_CRITIC_ENABLED=false uv run python -m evaluation.run_eval --stub
```

Also still 100%. The stub never proposes anything worth vetoing, so removing its
reviewer costs nothing. To see the Critic earn its keep you need a real model -
which is the honest reason this project keeps a live evaluation at all.

And the tests, which are the claim that the guardrails are real:

```bash
uv run pytest -q                              # the whole suite, no key, no network
uv run pytest tests/test_halt_conditions.py -v   # every ceiling, driven to a stop
uv run pytest tests/test_guardrails.py -v        # RBAC, risk, injection, redaction
```

---

## 3. The notebook

```bash
uv run jupyter lab ../notebooks/
```

Open `incident_commander_walkthrough.ipynb` and run it top to bottom. It runs each
agent in isolation, prints the **actual prompt text** each one receives, and lets
you attack the guardrail layer directly with your own strings. No key needed.

This is the demo for someone who wants to know *how* it works rather than *that*
it works.

---

## 4. The live console

This is the one to record. Real streaming, a real human approval gate, and (with a
key) real Gemini. Without a key it runs on the offline agent, with a banner saying so.

**Two terminals:**

```bash
# Terminal 1 - backend
cd backend
uv run uvicorn main:app --reload --port 8000

# Terminal 2 - frontend
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Or, with Docker: `docker compose up --build`, then
<http://localhost:8080>.

Check the top of the page first. An **"Offline mode"** banner means no key is
loaded: the agents are the rule-based offline agent, which is fine for showing the
loop and the gate but is not model reasoning. Say so out loud, or add the key and
restart ([ADR-0009](docs/adr/0009-gemini-mandatory-no-fallback.md)).

### The three-minute script

**Start with `bad_deploy`.** ~30s, ~$0.012. Narrate the stream as it arrives:
memory recall, triage, diagnosis with a confidence number, one proposed action from
a closed enum carrying a **falsifiable prediction**, the Critic arguing against it,
then the loop stopping dead at the human gate. Approve it. Verification pulls fresh
telemetry and checks whether the predicted signal actually moved.

**Then `poisoned_telemetry`.** The log lines contain attacker text demanding a
destructive failover. Open the telemetry panel and show that the demand has been
*removed* and replaced with an audit marker before any agent saw it - not merely
flagged. The agent restarts the service, which is the correct fix, and the guardrail
flags record the attempt.

**Then `unfixable_ca_expiry`.** Nothing in the action menu can fix an expired CA.
The correct behaviour is to escalate, and it does. This is the moment that
distinguishes an agent from a demo: it is willing to return nothing.

**If you have another minute, `cascading_failure`.** Two ordered fixes, and the
telemetry genuinely changes after the first one, so the loop re-diagnoses rather
than reusing a stale conclusion.

### Things worth clicking

- **Cancel**, mid-run. It is a real halt condition, not a UI state.
- **Reject** at the approval gate instead of approving. The incident ends
  `rejected_by_human`, and the reason is carried in the postmortem.
- The **cost panel**, which breaks spend down per agent, not just per incident.
- `http://localhost:8000/api/config` - every limit the agent is running under,
  inspectable at runtime rather than only documented.
- `http://localhost:8000/api/memory` - the episodes it has stored. Run the same
  scenario twice and the second run recalls the first.

### Keeping it cheap

One scenario at a time. The defaults are already the cheap ones (fast-tier model
on, LLM Supervisor off) - see [Cost and runtime](docs/testing-and-evaluation.md#cost-and-time). The
per-incident hard ceiling is $0.50 and is enforced twice, so a runaway demo is
bounded even if you walk away from it.

Do **not** open with a bare `run_eval` (no `--stub`, no `--scenarios`). That is ten
live scenarios, ~10 minutes and ~$0.22, and it is a measurement tool rather than a
demo.

If you have `make` available, `make demo` and `make demo-live` are the first and
last demos on this page. Every command here works without it.

---

## 5. Persistence and login

Optional, and worth showing precisely because the project works without it.

```bash
docker compose up -d postgres     # host port 5434
```

Then in `backend/.env`:

```bash
IC_DATABASE_URL=postgresql+asyncpg://incident:incident@localhost:5434/incident_commander
IC_AUTH_REQUIRED=true
IC_JWT_SECRET=<paste the output of: python -c "import secrets; print(secrets.token_urlsafe(32))">
IC_SEED_ADMIN_USER=admin
IC_SEED_ADMIN_PASSWORD=pick-something-better   # or leave empty: a random one is printed in the log
```

Restart the backend. `GET /api/health` now reports `db_connected: true` and
`login_required: true`, and the console redirects to a login page.

### The three things to show

**Failed logins are indistinguishable.** A wrong password and a username that does
not exist return the identical message, so the form cannot be used to enumerate
accounts:

```bash
curl -s -XPOST localhost:8000/api/auth/login -H 'Content-Type: application/json'   -d '{"username":"ghost","password":"nope"}'      # incorrect username or password
curl -s -XPOST localhost:8000/api/auth/login -H 'Content-Type: application/json'   -d '{"username":"admin","password":"nope"}'      # ...the same
```

**Incidents survive a restart.** Run one, kill the backend, start it again, and the
run is still in the list with its full event timeline and postmortem - reloaded
from Postgres rather than from process memory.

**Roles are enforced on the server.** A `viewer` can watch an incident but gets a
403 from `/approve`. The UI disables the button too, but that is a courtesy; the
control is `require_role` in `main.py`. Prove it by calling the API directly with a
viewer's token.

Worth saying out loud while demoing: the application role is **not** the agent's
role. Signing in as an admin does not widen what the agent may execute -
`IC_AGENT_ROLE` still governs that, and an agent set to `read-only` stays read-only
no matter who is logged in.

### What this auth layer is not

Real bcrypt with per-password salts and real signature verification, but no refresh
rotation, no lockout after repeated failures, no MFA, and no revocation list - a
signed-out token stays valid until it expires. Those gaps are named in
`backend/src/auth.py`. It is a demonstration of the shape, not something to put on
the internet as-is.

---

## Talking points, if you are being asked questions

The [interview guide](docs/interview-guide.md) answers the seven
questions this project usually gets. The one-line version:

> The model decides what to do. Code decides what is allowed.

Every guardrail worth claiming is enforced outside the model and has a test that
drives it to a real stop. The failures in the evaluation are left in on purpose: an
eval that always passes is measuring nothing.
