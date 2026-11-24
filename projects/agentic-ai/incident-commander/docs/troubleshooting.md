# Troubleshooting

Problems people actually hit when running this for the first time, and the fix for
each. Back to the [README](../README.md).

## Setup

**`uv: command not found`**
Install uv: `pip install uv`, or follow
<https://docs.astral.sh/uv/getting-started/installation/>. Open a new terminal after
installing so it is on your `PATH`.

**`npm install` fails, or Vite says Node is too old**
The console needs **Node.js 22.12 or newer** (24 LTS recommended). Check with `node --version`. If you use
nvm, `nvm use` in `frontend/` picks the version in `.nvmrc`.

**`make: command not found` (Windows)**
`make` is optional. Every Make target is a plain command; run the commands from the
README instead.

**Environment variables in examples like `IC_ENGINE=graph uv run ...` do not work
(Windows PowerShell)**
That syntax is for bash and zsh. In PowerShell set the variable first:

```powershell
$env:IC_ENGINE = "graph"; uv run pytest -q
Remove-Item Env:IC_ENGINE        # undo it afterwards
```

In `cmd.exe` use `set IC_ENGINE=graph` and then the command.

## The backend

**Port 8000 is already in use** (`error while attempting to bind`)
On Windows there may be no error at all: two programs can share the port, and a
different app answers. Check http://localhost:8000/api/health shows
`"role": "sre-bot"`. If it does not, run the backend on another port and tell the
console where it is:

```bash
uv run uvicorn main:app --port 8010                 # terminal 1
IC_BACKEND_URL=http://localhost:8010 npm run dev    # terminal 2 (bash)
```

```powershell
$env:IC_BACKEND_URL = "http://localhost:8010"; npm run dev    # terminal 2 (PowerShell)
```

If port 5173 is taken, Vite picks the next free port and prints it; open that one.

**I set a key, but the console still says "Offline mode"**
- The key must be in `backend/.env` (or a `.env` next to `docker-compose.yml` for
  Docker), on a line like `GOOGLE_API_KEY=AIza...`. `GEMINI_API_KEY` also works.
- Restart the backend after editing `.env`; it is read at startup.
- `GET http://localhost:8000/api/health` shows `llm_mode` and a `detail` line that
  says which part is missing.
- `IC_LLM_MODE=offline` forces offline mode even with a key. Remove it.

**Incidents are refused with HTTP 503**
You set `IC_LLM_MODE=gemini` (Gemini only) but there is no key. Add the key, or
remove `IC_LLM_MODE` to allow offline mode.

**A live run fails with "model not found" or 404**
Google retires and restricts model versions. The defaults use Google's stable
aliases (`gemini-flash-latest`, `gemini-flash-lite-latest`). If you pinned a version
with `IC_MODEL_NAME` / `IC_MODEL_FAST`, check it is still listed at
<https://ai.google.dev/gemini-api/docs/models>.

**A live run stops with `llm_unavailable`**
The model call failed after retries (network, quota, or rate limit). The incident is
escalated rather than guessed, and still gets a postmortem, assembled from the incident
record (the failed model is not asked again). Check the backend log for
the underlying error; free-tier keys hit per-minute limits quickly.

## The console

**The page loads but nothing happens when I click Trigger Incident**
The backend is not reachable. The header chip says **Disconnected**. Start the
backend (terminal 1), or see "Port 8000 is already in use" above.

**An incident sits at "awaiting approval"**
That is the human gate working. Scroll to the approval card and click **Approve**
or **Reject**. If nobody answers, it escalates after `IC_APPROVAL_TIMEOUT_S`
(300 s by default).

**`Cannot find module @rollup/rollup-<platform>`**
A known npm bug with optional native packages. Delete `node_modules` and run
`npm install` again (keep `package-lock.json`).

## Docker

**`docker compose up` cannot reach the backend, or ports are taken**
Compose publishes 8080 (console), 8000 (API) and 5434 (Postgres) on `127.0.0.1`.
Stop whatever owns them, or pick other ports for this run:

```bash
IC_CONSOLE_HOST_PORT=8081 IC_API_HOST_PORT=8011 POSTGRES_HOST_PORT=5435 docker compose up --build
```

```powershell
$env:IC_CONSOLE_HOST_PORT = "8081"; $env:IC_API_HOST_PORT = "8011"; docker compose up --build
```

Then open the console port you chose (http://localhost:8081 above).
Set `POSTGRES_HOST_PORT` to move Postgres.

**I turned on login and do not know the admin password**
If `IC_SEED_ADMIN_PASSWORD` was empty, a random password was generated and printed
once in the backend log: `docker compose logs backend | grep "seeded admin"`.
See [persistence-and-login.md](persistence-and-login.md).

## Tests

**Tests try to call Gemini**
They should never do that. The test setup clears any key from the environment
before the app loads. If you see network calls, you are probably running `pytest`
from outside `backend/`; `cd backend` first.

**`npm test` fails with "Vitest failed to find the runner" (Windows)**
`npm test` already works around this (a terminal opened in `c:\...` with a lowercase
drive letter makes Vitest load itself twice). If you run `npx vitest` directly, open
the terminal in `C:\...` instead, or use `npm test`.
