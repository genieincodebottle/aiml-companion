# Hermes Ops Agent

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Hermes Agent](https://img.shields.io/badge/Hermes_Agent-MIT-green.svg)
![No API key to start](https://img.shields.io/badge/Demo-No_API_key-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Operate [Hermes Agent](https://github.com/NousResearch/hermes-agent) properly, and
**measure whether its learning loop actually pays off** on your own machine.

Hermes claims that once it has written a skill for a task, doing that task again
costs less. This project checks that claim against Hermes' own SQLite store,
ships the skills and gateway hardening to run it for real, and includes a
repeatable prompt-injection probe.

```
Run 1 (cold)          Run 2 (skill on disk)        What the harness reports
web_search x2         skill_view  x1               tool_calls   11 ->  4
read_file   x3        terminal    x3               billable     51,300 -> 11,400
terminal    x5                                     total_tokens 51,300 -> 42,400  <- misleading
skill_manage x1                                    VERDICT: skill shortened the procedure
```

## Quick Start: no API key, no Hermes install

The measurement runs against a recorded fixture, so you can see it work before
signing up for anything.

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/hermes-ops-agent

python main.py demo          # the measurement, on recorded data
python -m pytest tests/ -q   # 65 tests, no network
```

There is no setup step and nothing to install first. The demo database is
built automatically on first run.

`make` is a convenience wrapper, not a requirement, and it is usually absent
on Windows. Every target has a plain-Python equivalent:

| Make | Without make |
|---|---|
| `make demo` | `python main.py demo` |
| `make test` | `python -m pytest tests/ -q` |
| `make sessions` | `python main.py sessions` |
| `make compare BASE=x CAND=y` | `python main.py compare x y` |
| `make probe` | `python main.py probe --write ./probe` |
| `make install-skills` | copy `skills/` into `~/.hermes/skills/` |
| `make fixture` | not needed; it builds itself |

On macOS and most Linux distributions the command is `python3`, not `python`.

The harness itself is **stdlib only**: `sqlite3`, `argparse`, `dataclasses`.
`requirements.txt` exists for pytest and a YAML config check.

## Then: against your own agent

### 0. What this costs

Hermes is free. You pay whoever serves the model, and an agent loop makes many
calls per task, so this is worth setting before you start rather than after.

- **Nous Portal** (`hermes setup --portal`) and the **Gemini free tier** both
  have free allowances that cover the exercises here.
- A paid model on a long task with a dead end in it can run to a few dollars.
  The whole point of step 4 is that the task is long.
- `hermes -z "..." --usage-file report.json` writes `estimated_cost_usd` for a
  single run, and `hermes sessions stats` totals your store. Look at both once
  early, before you assume the cost is small.

### 1. Install Hermes and pick a model

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes setup --portal      # OAuth, no key to paste
# or: hermes model         # Gemini / OpenRouter / OpenAI / Anthropic / local
```

**Hermes needs a model with at least 64,000 tokens of context.** The prompt
carries identity, context files, skill metadata, memory and tool schemas
before your message is appended.

A note on local models: Ollama works and costs nothing, but the exercise below
is a long multi-step task with a dead end in it, which is the hardest thing to
ask of a model. Small local models fail at sustained tool calling, and they
fail by emitting malformed tool calls. Start on a hosted model, move local once
you know what a working run looks like.

### 2. Use the hardened config

```bash
cp configs/config.yaml ~/.hermes/config.yaml
```

`write_approval` on for both memory and skills, compression pointed at a cheap
model, tool output capped, and a Docker terminal backend.

**If you do not have Docker installed, change one line first**, or every shell
command the agent tries will fail:

```yaml
terminal:
  backend: local     # instead of: docker
```

`local` means the agent runs commands directly on your machine with no
sandbox. That is fine while you are following this project on a scratch
directory, and it is not fine once the agent is reading your real files or
running unattended. Install Docker and switch back before step 6.

### 3. Install the skill library

```bash
# macOS / Linux
mkdir -p ~/.hermes/skills && cp -R skills/. ~/.hermes/skills/
# Windows PowerShell
#   Copy-Item -Recurse skills\* $env:USERPROFILE\.hermes\skills\
# or, if you have make:  make install-skills

hermes skills browse
```

Two hand-written skills (`cut-release-candidate`, `restore-staging-db`) in the
[agentskills.io](https://agentskills.io/specification) format. Read them before
you let the agent write its own; the four sections each do a job.

### 4. Run the experiment

If you have a repetitive task of your own with a dead end in it, use that. If
you do not, this one works on any machine and takes a few minutes:

> In a scratch directory, find every file larger than 1 MB that has not been
> modified in 90 days, write the list to `old-files.txt` sorted by size, and
> tell me the total size reclaimed if I deleted them. Do not delete anything.

It is a good test task because it has a real dead end: `find -mtime` and
`stat` differ between macOS/BSD and Linux, so the agent usually gets the flags
wrong once, sees the error, and corrects. That correction is exactly what a
skill is supposed to capture.

```bash
# Cold run. Fresh session, no matching skill on disk.
hermes
#   > <the task above, or your own>
#   ... let it flail, recover, finish
#   > save that as a skill so you do not repeat the detour

# Fresh session. Same task, worded the same way.
hermes
#   > <the same task>

python main.py sessions                    # pick the two ids
python main.py compare sess_xxx sess_yyy
```

**Two things that will otherwise make the comparison meaningless**, and which
the tool warns you about rather than silently averaging away:

- The second run has to be a **fresh session**, not `hermes --continue`. A
  continuation already has the first run in its context.
- Both runs need the **same model**. Switching models between runs measures
  the model, not the skill.

### 5. Probe your own agent

```bash
python main.py probe --write ./probe       # writes the payload files
hermes --in ./probe
#   > summarise each file in this directory
python main.py probe --session <session-id>
```

Three payload shapes. None of them exfiltrate anything: they ask for a canary
string and a variable *name*. If the canary comes back, a real payload would
have worked.

### 6. Deploy a gateway you could leave running

```bash
sudo cp deploy/hermes-gateway.service /etc/systemd/system/
sudo systemctl enable --now hermes-gateway
bash deploy/verify-allowlist.sh @your_bot
```

`verify-allowlist.sh` checks that `GATEWAY_ALLOW_ALL_USERS` is unset, that an
explicit allowlist exists, and that `.env` is mode 600 - then makes you do the
one check that actually proves it: message the bot from an account that is not
on the list.

## Why the numbers are reported the way they are

A second run is cheaper for two reasons, and only one of them is the skill:

1. The skill made the procedure shorter.
2. The prompt prefix was cached.

`total_tokens` mixes them and will show a large "improvement" on a run where
the skill never loaded. So **`tool_calls` is the headline metric** - caching
cannot remove a tool call - and the cache columns are printed next to it rather
than hidden.

Full reasoning, including the provider-dependent caveat on `input_tokens` vs
`cache_read_tokens`, is in [docs/MEASUREMENT.md](docs/MEASUREMENT.md).

## Layout

```
main.py                     CLI: sessions | compare | demo | probe
src/fixture.py              builds the demo database on first run
src/state_db.py             read-only reader for ~/.hermes/state.db
src/metrics.py              deltas, the cache confound, the warnings
src/report.py               terminal and JSON rendering
src/injection.py            prompt-injection payloads and canary scan
skills/                     installable SKILL.md library
configs/config.yaml         hardened Hermes config
deploy/                     systemd unit, compose file, allowlist check
fixtures/state.fixture.sql  the demo data, as reviewable SQL
docs/MEASUREMENT.md         why the measurement is shaped this way
```

## Safety

The reader opens `state.db` with `mode=ro`. That file is your live agent
memory, and SQLite will happily let a tool corrupt it otherwise. A test asserts
the connection refuses writes and that the file is byte-identical after a full
read.

Nothing in this project writes to `~/.hermes` except the skill-install step,
which only copies files into `skills/`.

## Requirements

- Python 3.11+
- Hermes Agent (only for steps 1-6; the demo and tests need neither)
- Docker, for the recommended terminal backend and the compose deployment

## License

MIT, matching Hermes Agent.
