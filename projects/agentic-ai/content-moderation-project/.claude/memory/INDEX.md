# Memory index

One line per fact. Loaded at the start of each session so a cold start knows what already exists. Keep it short; the detail lives in the linked file.

- [Model tiering decision](decision-model-tiering.md) - why search and verify use cheap models and only generation uses the top model.
- [Gotcha: never commit databases or .env](gotcha-never-commit-databases-or-env.md) - local state and secrets are gitignored; never stage them.