---
name: reset-data
description: Clean all claims/HITL data and re-seed policies for a fresh start
disable-model-invocation: true
allowed-tools: Bash(python *)
---

# Reset Project Data

Run a full data reset and re-seed:

```bash
python scripts/clean_data.py
python scripts/seed_policies.py
```

If `$ARGUMENTS` contains "all" or "full", do a complete reset including users:

```bash
python scripts/clean_data.py --all
python scripts/seed_policies.py
```

Report what was cleaned and re-seeded. Warn that if `--all` was used, the user will need to restart the backend to re-create seed accounts.