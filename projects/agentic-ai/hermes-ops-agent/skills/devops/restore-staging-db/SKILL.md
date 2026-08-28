---
name: restore-staging-db
description: Restore the staging Postgres from the latest nightly dump, including the VPN requirement and the 5433 port quirk
version: 1.0.0
platforms: [macos, linux]
metadata:
  hermes:
    tags: [postgres, staging, backup]
    category: devops
---

## When to Use

The user asks to reset, refresh or restore the staging database, or reports
staging data that is stale or corrupted.

## Procedure

1. Confirm the VPN is up. Nothing below works without it, and the failure is
   a silent timeout rather than an error.

       nc -z db.staging.internal 5433

2. Find the newest dump.

       ls -t /backups/staging/*.dump | head -1

3. Terminate open connections, or the drop hangs indefinitely.

       psql -h db.staging.internal -p 5433 -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='staging';"

4. Restore.

       pg_restore -h db.staging.internal -p 5433 -d staging --clean --if-exists <dump>

## Pitfalls

- The port is 5433, not 5432. 5432 is the local dev instance, and connecting
  to it SUCCEEDS against the wrong database.
- `--clean` without `--if-exists` fails on a fresh database.
- Never run this against a host without `staging` in the name.

## Verification

    psql -h db.staging.internal -p 5433 -d staging -c "SELECT count(*) FROM users;"

Should return a count within about 5% of production.
