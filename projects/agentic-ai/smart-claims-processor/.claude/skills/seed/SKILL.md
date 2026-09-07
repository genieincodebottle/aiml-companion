---
name: seed
description: Seed test policies (US + India) with valid dates into the database
disable-model-invocation: true
allowed-tools: Bash(python *)
---

# Seed Test Policies

Run the policy seeder to populate both US and India test policies:

```bash
python scripts/seed_policies.py
```

Report what was seeded. If the backend is running, mention that new claims can now reference these policies.