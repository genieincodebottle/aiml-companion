# Troubleshooting

## Setup

**`uv: command not found`**
Install uv: https://docs.astral.sh/uv/getting-started/installation/ (one command on
macOS, Linux and Windows), then open a new terminal.

**`npm ci` fails with "Missing ... from lock file" or an engine warning**
The console needs Node 22.12 or newer (`node --version`). With nvm: `nvm use` in
`frontend/` picks the version in `.nvmrc`.

**`npm ci` on Windows fails with `Cannot find module @rollup/rollup-win32-x64-msvc`**
A known npm optional-dependency bug. Delete `frontend/node_modules` and run `npm ci`
again.

**Port 8000 is already in use**
On Windows there may be no error at all: two programs can share the port, and a
different app answers. Check http://localhost:8000/api/health shows `"can_run"`.
If it does not, run the backend on another port (`uv run uvicorn main:app --port 8010`) and start the
console with `MC_BACKEND_URL=http://localhost:8010 npm run dev`
(PowerShell: `$env:MC_BACKEND_URL="http://localhost:8010"; npm run dev`).

## The console

**The header says "backend down"**
The backend is not running or not on port 8000. Start it (terminal 1 of the Quick
Start) and reload.

**"Offline mode" although I set a key**
The key must be in `backend/.env` (or exported in the backend's terminal) as
`GOOGLE_API_KEY=...`, and the backend restarted. `GET /api/health` shows `mode` and a
`detail` line saying what it found.

**The Approve button is greyed out**
You opened the case, so maker-checker stops you deciding it. Switch **Acting as** to
`rex.reviewer` (or sign in as a reviewer).

**"a valid operator token is required"**
`MC_OPERATORS` is set, so login is on. Sign in with a token from that list, or unset
it for local use. See [operators-and-approvals.md](operators-and-approvals.md).

## Live runs

**A case ends `dead_lettered` with "all providers failed"**
Both models failed twice: usually a wrong key, a quota limit or no network. The reason
is on the case. Check the backend log for the provider's own message, and the key at
https://aistudio.google.com/apikey.

**A case ends `dead_lettered` with "budget exceeded"**
It hit `MC_MAX_COST_USD`. A live case costs about $0.004, so the default $0.25 only
trips if something loops; raise it only if you changed the models.

**`503 MC_MODE=live but GOOGLE_API_KEY is not set`**
You asked for live mode and there is no key. Set the key, or use `MC_MODE=auto`.

**A live result differs from the last run**
Models are not deterministic. Run the eval with `--repeats 3` and read the rate.

A live `--repeats 3` run takes about 10 minutes and prints the table only at the end,
so a quiet terminal is normal. One live case can end resolved one time and escalated
the next; judge the model by the rate over repeats.

## Windows

The `VAR=1 command` form in the docs is bash. In PowerShell:

```powershell
$env:MC_MODE="offline"; uv run python cli.py run ALRT-001
Remove-Item Env:MC_MODE
```

Red text from `npm` or `uvicorn` in PowerShell is usually a warning or an INFO log on
stderr, not an error. Read the words before stopping.

`make` is not installed on Windows by default; every Makefile target is one line you
can run directly.
