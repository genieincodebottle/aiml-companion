# Persistence and login (both optional)

By default the project needs **no infrastructure**: incidents live in memory and
runbook memory is a JSON file. That is a fully supported setup, not a degraded one.
Back to the [README](../README.md).

Adding Postgres gives you three things:

| With no database | With Postgres |
|---|---|
| Incidents vanish on restart | Incidents, events and postmortems survive |
| The dashboard counts what is in memory | Counts every run ever, computed in SQL |
| No users, no login | `IC_AUTH_REQUIRED=true` puts the console behind a login |

## Turn on Postgres

With Docker Compose it is already on: `docker compose up --build` starts Postgres and
points the backend at it.

For the two-terminal setup, start only the database and point the backend at it:

```bash
docker compose up -d postgres     # listens on 127.0.0.1:5434
```

Then in `backend/.env`:

```
IC_DATABASE_URL=postgresql+asyncpg://incident:incident@localhost:5434/incident_commander
```

Port 5434, not 5432, because a local Postgres install often owns 5432 and the
resulting "password authentication failed" error is baffling.

## Schema and migrations

The schema is managed by [Alembic](https://alembic.sqlalchemy.org/). The backend
**applies pending migrations itself at startup**, so there is no separate step. A
database created by an older version of this project (before migrations existed)
is adopted automatically: it is stamped at the first migration, not rebuilt, and its
data is kept.

When you change `backend/src/db/models.py`:

```bash
cd backend
uv run alembic revision --autogenerate -m "add approver column"   # write the migration
# read the generated file in migrations/versions/ - autogenerate is a draft, not a promise
uv run alembic upgrade head                                       # or just restart the backend
uv run alembic check                                              # "No new upgrade operations detected."
```

`uv run alembic upgrade head --sql` prints the SQL without touching a database. The
test suite uses that to check every table and index in the models has a migration.

## Turn on login

Login needs the database (users have to live somewhere). In `backend/.env` (or the
`.env` next to `docker-compose.yml`):

```
IC_AUTH_REQUIRED=true
IC_JWT_SECRET=<a random value, at least 32 characters>
```

Generate the secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The placeholder secret is refused, because a known signing key lets anyone mint an
admin token.

**The first admin.** The first time the backend starts with login on and an empty
users table, an admin is created:
username `IC_SEED_ADMIN_USER` (default `admin`), password `IC_SEED_ADMIN_PASSWORD`.
If you leave the password empty (the default), a random one is **generated and
printed once** in the backend log:

```bash
docker compose logs backend | grep "seeded admin"     # Docker
# or read the terminal running uvicorn
```

There is no fixed default password, on purpose. Further accounts are created by an
admin through `POST /api/auth/register`.

## Roles

| Role | May |
|---|---|
| `viewer` | watch incidents, read history and memory |
| `operator` | also start incidents, approve, reject and cancel |
| `admin` | also create accounts |

An unknown role may do nothing. These are the roles of the **people** using the
console. They do not change what the **agent** may execute: that is `IC_AGENT_ROLE`,
and an agent set to `read-only` stays read-only no matter who is logged in.

## How the login works

Passwords are hashed with bcrypt (72-byte limit, enforced). Login returns a signed
token, used as a `Bearer` header by the console, and also sets it as an HttpOnly,
SameSite=Strict cookie. The cookie is what lets the browser's live event stream
(which cannot send headers) require a login like every other endpoint.

It is deliberately a demonstration, and the gaps are named in
[`backend/src/auth.py`](../backend/src/auth.py): no refresh-token rotation, no
lockout after failed attempts, no MFA, and no revocation list, so a token stays
valid until it expires even after logout.

## Exposing it beyond your machine

Everything binds to `127.0.0.1` by default, and Docker Compose publishes its ports on
`127.0.0.1` only. Before you change that, turn login on: with login off, anyone who
can reach the API can approve actions.
