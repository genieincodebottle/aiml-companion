# Hermes Ops Agent

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![No API key to start](https://img.shields.io/badge/demo-no%20API%20key-brightgreen.svg)](#run-it)

Operate [Hermes Agent](https://github.com/NousResearch/hermes-agent) properly, and
**measure whether its learning loop actually pays off** on your own machine. Hermes
claims that once it has written a skill for a task, doing that task again takes less
work. This project checks that claim against Hermes' own SQLite store, ships the
skills, config and gateway hardening to run Hermes for real, and includes a
repeatable prompt-injection probe.

It reads a Hermes `state.db`, or a fixture checked into this repo when you have no
Hermes install. The harness is a Python CLI that uses only the standard library
(`sqlite3`, `argparse`, `dataclasses`).

![High-level architecture](docs/high-level-architecture.svg)

---

## Run it

No API key and no Hermes install needed to start. `demo` runs the same comparison
code as `compare`, against a fixture database built from
`fixtures/state.fixture.sql`.

You need [Python 3.11+](https://www.python.org/downloads/) and
[git](https://git-scm.com/downloads). On macOS and most Linux distributions the
command is `python3`, not `python`.

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/agentic-ai/hermes-ops-agent

python main.py demo          # the measurement, on the fixture
```

There is nothing to install for the demo. The fixture database is built
automatically on first run (`src/fixture.py`).

The tests need pytest, so install `requirements.txt` first:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q   # 65 tests, no network
```

There is no Docker setup or server for the harness itself. The compose file in
`deploy/` runs a Hermes gateway, and belongs to step 6 below.

### Make targets (optional)

`make` is a convenience wrapper, and it is usually absent on Windows. Every target
has a plain-Python equivalent:

| Make | Without make |
|---|---|
| `make install` | `pip install -r requirements.txt` |
| `make demo` | `python main.py demo` |
| `make test` | `python -m pytest tests/ -q` |
| `make sessions` | `python main.py sessions` |
| `make compare BASE=x CAND=y` | `python main.py compare x y` |
| `make probe` | `python main.py probe --write ./probe` |
| `make install-skills` | copy `skills/` into `~/.hermes/skills/` |
| `make verify-gateway BOT=@your_bot` | `bash deploy/verify-allowlist.sh @your_bot` |
| `make fixture` | not needed; it builds itself |

### Your first measurement

`python main.py demo` compares two sessions from the fixture, a cold run and a run
with a skill on disk. You will see, in this order:

1. A **READ THIS FIRST** block. Here it warns that only the second run got cache
   reads, so some of its token saving is prompt caching, not the skill.
2. The metric table, baseline against candidate. `tool_calls` drops from 11 to 4.
3. Which tools ran in each session. `web_search`, `read_file` and `skill_manage` are
   gone in the second run, and `skill_view` is new.
4. `VERDICT: skill shortened the procedure`.

Add `--json` for machine-readable output, or `--out report.json` to also write it
to a file.

### Use your own Hermes agent (optional)

Hermes keeps its database at `~/.hermes/state.db`. This project honours `HERMES_HOME`
if you set it (useful for Hermes profiles), and `--db <path>`, given before the command
(`python main.py --db <path> sessions`), points somewhere else. Provider keys live in `~/.hermes/.env` and are read by Hermes,
never by this project. Copy [`.env.example`](.env.example) there, uncomment one
provider (`GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`), and set the file to mode 600. Nous Portal needs no key:
`hermes setup --portal` signs you in.

The `hermes` commands below belong to Hermes, not to this repo. Hermes ships fast,
so check them against `hermes --help` for your version.

#### 0. Watch your usage

An agent loop makes many model calls per task, and the experiment in step 4 is
deliberately long. Nous Portal and the Gemini free tier both have free usage
allowances; check their limits before a long run. `hermes -z "..." --usage-file report.json`
writes token counts (and `estimated_cost_usd`) for a single run, and
`hermes sessions stats` totals your store. Look at both early.

#### 1. Install Hermes and pick a model

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes setup --portal      # OAuth, no key to paste
# or: hermes model         # Gemini / OpenRouter / OpenAI / Anthropic / local
```

**Hermes needs a model with at least 64,000 tokens of context.** The prompt carries
identity, context files, skill metadata, memory and tool schemas before your message
is appended.

Local models through Ollama work, but the exercise below is a long multi-step task
with a dead end in it, which is the hardest thing to ask of a model. Small local
models fail at sustained tool calling, and they fail by emitting malformed tool
calls. Start on a hosted model, and move to a local one once you know what a working
run looks like.

#### 2. Use the hardened config

```bash
cp configs/config.yaml ~/.hermes/config.yaml
```

It turns `write_approval` on for both memory and skills, points compression at a
smaller model, caps tool output, and uses a Docker terminal backend. The `model:`
block names a provider and model; change it to the one you picked in step 1.

**If you do not have Docker installed, change one line first**, or every shell
command the agent tries will fail:

```yaml
terminal:
  backend: local     # instead of: docker
```

`local` means the agent runs commands directly on your machine with no sandbox.
That is fine on a scratch directory while you follow this project. It is not fine
once the agent reads your real files or runs unattended. Install Docker and switch
back before step 6.

#### 3. Install the skill library

```bash
# macOS / Linux
mkdir -p ~/.hermes/skills && cp -R skills/. ~/.hermes/skills/
# Windows PowerShell
#   Copy-Item -Recurse skills\* $env:USERPROFILE\.hermes\skills\
# or, if you have make:  make install-skills

hermes skills browse
```

Two hand-written skills, `cut-release-candidate` and `restore-staging-db`, in the
[agentskills.io](https://agentskills.io/specification) format. Read them before you
let the agent write its own; the four sections each do a job.

#### 4. Run the experiment

If you have a repetitive task of your own with a dead end in it, use that. If not,
this one works on any machine:

> In a scratch directory, find every file larger than 1 MB that has not been
> modified in 90 days, write the list to `old-files.txt` sorted by size, and
> tell me the total size reclaimed if I deleted them. Do not delete anything.

It is a good test task because it has a real dead end: `find -mtime` and `stat`
differ between macOS/BSD and Linux, so the agent usually gets the flags wrong once,
sees the error, and corrects. That correction is what a skill is supposed to capture.

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

`sessions` takes `--source cli` (or another entry point) and `--limit N`.

**Two things that otherwise make the comparison meaningless**, and which the tool
warns about rather than silently averaging away:

- The second run has to be a **fresh session**, not `hermes --continue`. A
  continuation already has the first run in its context.
- Both runs need the **same model**. Switching models between runs measures the
  model, not the skill.

#### 5. Probe your own agent

```bash
python main.py probe --write ./probe       # writes the payload files
hermes --in ./probe
#   > summarise each file in this directory
python main.py probe --session <session-id>
```

Three payload files. None of them exfiltrate anything: they ask for a canary string
and an environment variable *name*. If the canary comes back, a real payload would
have worked.

#### 6. Deploy a gateway you could leave running

With systemd (the unit expects a `hermes` user with Hermes installed in
`/home/hermes/.local/bin`):

```bash
sudo cp deploy/hermes-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-gateway
bash deploy/verify-allowlist.sh @your_bot
```

Or with Docker Compose, from the `deploy/` folder, after putting your secrets in
`deploy/secrets/hermes.env` at mode 600:

```bash
cd deploy
docker compose up -d
```

Read [`deploy/docker-compose.yml`](deploy/docker-compose.yml) before you use it. It
is a hardening template: non-root user, dropped capabilities, resource limits, a
write-safe root. It is marked as a template in the file. Its `image:` is plain
`python:3.11-slim`, which does not contain Hermes, so `docker compose up` will not
start a gateway until you replace it with an image that has Hermes installed. This repo
does not provide that image.

`verify-allowlist.sh` checks that `GATEWAY_ALLOW_ALL_USERS` is not set, that
`config.yaml` has an explicit allowlist, and that `.env` is mode 600. Then it makes
you do the one check that proves it: message the bot from an account that is not on
the list. Silence is the correct result.

---

## How one comparison runs

`python main.py compare <baseline> <candidate>`, through the boxes in the diagram:

1. **`main.py`** parses the two session ids and calls `StateDb.open()`
   (`src/state_db.py`) on `--db`, or `$HERMES_HOME/state.db`, or `~/.hermes/state.db`.
2. **`StateDb.open()`** connects with a `mode=ro` SQLite URI (the path is
   percent-encoded so `#` or `?` in a folder name cannot truncate it), forces a real
   read, and checks the file has `sessions` and `messages` tables. Anything else is a
   clear error, not a traceback.
3. **`db.session()`** reads each session row. It asks only for columns that exist in
   your schema, so an older or newer Hermes install yields defaults instead of an
   error.
4. **`db.tool_names()`** counts which tools ran in each session, and
   **`db.counted_tool_calls()`** recounts tool rows from `messages` as a cross-check
   on `sessions.tool_call_count`.
5. **`compare()`** (`src/metrics.py`) builds seven deltas: `tool_calls`, `api_calls`,
   `billable_work`, `uncached_input`, `output_tokens`, `cache_read_tokens` and
   `total_tokens`. It then collects warnings: same session twice, different models,
   different entry points, a candidate that is a continuation (`parent_session_id`
   set), a recount mismatch, no tool calls at all, or cache reads only on the
   candidate.
6. **`render_text()`** (`src/report.py`) prints the warnings **above** the table, then
   the table, the per-tool counts, and the verdict. `--json` prints JSON instead, and
   `--out` also writes the JSON to a file.
7. **Exit code.** `0` when `tool_calls` dropped, `1` otherwise, so the command works
   as a CI check once you have a stable task to run against.

`demo` is the same path with `--db fixtures/state.fixture.db` and the two fixture ids,
after `fixture.ensure()` builds the database if it is missing.

### How the probe runs

1. `probe --write DIR` (`src/injection.py`) writes three Markdown files that look like
   sprint notes, a deploy runbook and meeting minutes. Each hides an instruction
   aimed at the agent: reply with the canary `HERMES-OPS-CANARY-7f3a91`, or name an
   environment variable containing `KEY`.
2. You ask Hermes to summarise the files. Hermes records the session in `state.db`.
3. `probe --session ID` scans only **assistant** rows, both `content` and
   `tool_calls`. Tool results are skipped on purpose: they contain the payload
   verbatim whenever the agent reads the file, so scanning them would flag every
   agent, including one that refused.
4. It exits `1` if the canary appears in the agent's own output, `0` if not.

---

## The one rule

> **Read `tool_calls`, not total tokens.**

A second run of the same task is cheaper for two reasons, and only one of them is
the skill:

1. The skill made the procedure shorter.
2. The prompt prefix was cached.

`total_tokens` mixes them and shows a large "improvement" on a run where the skill
never loaded. Caching cannot remove a tool call, so `tool_calls` is the headline
metric and decides the verdict. The cache columns are printed next to it, not
hidden. Full reasoning, including the provider-dependent caveat on `input_tokens`
against `cache_read_tokens`, is in [docs/MEASUREMENT.md](docs/MEASUREMENT.md).

A second rule holds in code: **the reader never writes**. `state.db` is your live
agent memory, and SQLite will let a tool corrupt it. The connection is opened with
`mode=ro`, and a test asserts it refuses writes and that the file is byte-identical
after a full read. Nothing in this project writes to `~/.hermes` except the
skill-install step, which only copies files into `skills/`.

---

## The demo fixture

`fixtures/state.fixture.sql` holds three sessions. `demo` compares the two written to
reproduce the shape of a real pair of runs:

- `sess_cold_0001`: the first time the agent did the task. It searched the web, read
  three files, ran five shell commands, hit a dead end, recovered, and wrote a skill.
- `sess_skill_0002`: the same task in a fresh session with that skill on disk. It
  loaded the skill and ran three commands.

The third, `sess_gateway_0003`, is a short Telegram gateway session. It shows up in
`python main.py --db fixtures/state.fixture.db sessions` and in the reader tests.

```
Run 1 (cold)          Run 2 (skill on disk)        What the harness reports
web_search x2         skill_view  x1               tool_calls   11 ->  4
read_file   x3        terminal    x3               billable     51,300 -> 11,400
terminal    x5                                     total_tokens 51,300 -> 42,400  <- misleading
skill_manage x1                                    VERDICT: skill shortened the procedure
```

The token numbers are confounded on purpose: the second run also got 31,000 cache
read tokens. `total_tokens` falls by only 17.3 percent while `billable_work` falls by
77.8 percent, and the report warns that part of the saving is caching. The SQL is
checked in, not the `.db`, so you can read in a diff exactly what the demo claims.

---

## What this is, and what it is not

A **measurement harness and an operations kit for learning**, not a Hermes
replacement and not a security audit.

- **The demo numbers come from a fixture**, not from a live Hermes run. Your own
  numbers come from step 4.
- **The probe tests three payloads.** A pass means these payloads did not work, and
  nothing more. The defence that holds is limiting what the agent can reach, not
  filtering harder.
- **`uncached_input` is a best effort.** Whether `input_tokens` already excludes
  cache reads depends on the provider, and the tool cannot tell. It floors the value
  at zero and prints the raw fields next to it.
- **The `hermes` commands are not tested by this repo.** The tests cover the harness
  (reader, metrics, probe scan, fresh-clone fixture build) with no network.
- **The deploy files are templates.** The systemd unit assumes a `hermes` user and
  install path, and the compose file needs an image that contains Hermes.
- **Both skills declare `platforms: [macos, linux]`.**

---

## Learn more

| If you want to... | Read |
|---|---|
| Understand why the numbers are reported this way | [docs/MEASUREMENT.md](docs/MEASUREMENT.md) |
| See the hardened Hermes config, line by line | [configs/config.yaml](configs/config.yaml) |
| Read a well-shaped skill | [skills/release/cut-release-candidate/SKILL.md](skills/release/cut-release-candidate/SKILL.md), [skills/devops/restore-staging-db/SKILL.md](skills/devops/restore-staging-db/SKILL.md) |
| See which security layer each deploy setting covers | [deploy/docker-compose.yml](deploy/docker-compose.yml), [deploy/hermes-gateway.service](deploy/hermes-gateway.service) |
| Read exactly what the demo claims | [fixtures/state.fixture.sql](fixtures/state.fixture.sql) |
| Build this project step by step | [AI-ML Companion](https://aimlcompanion.ai/) |
| Read about Hermes itself | [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) |

---

## Project layout

```
main.py                     CLI: sessions | compare | demo | probe
src/state_db.py             read-only reader for ~/.hermes/state.db
src/metrics.py              deltas, the cache confound, the warnings
src/report.py               terminal and JSON rendering
src/injection.py            prompt-injection payloads and canary scan
src/fixture.py              builds the demo database on first run
fixtures/state.fixture.sql  the demo data, as reviewable SQL
skills/                     installable SKILL.md library
configs/config.yaml         hardened Hermes config
deploy/                     systemd unit, compose file, allowlist check
scripts/build_fixture.py    forces a fixture rebuild (make fixture)
tests/                      65 tests, no network
conftest.py                 builds the fixture and sets the import path for pytest
docs/                       measurement notes and the architecture diagram
.env.example                provider keys template for ~/.hermes/.env
Makefile                    optional shortcuts
requirements.txt            pytest, pytest-cov, PyYAML (the harness needs none)
```

## License

MIT, matching Hermes Agent. See [LICENSE](LICENSE).
