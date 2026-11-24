# Candidate prompts

Each file maps a prompt name to a new template. Score one with:

```bash
uv run python -m evaluation.run_eval --candidate evaluation/candidates/narrative_judgement.json
```

The candidate is registered under the `candidate` label and every golden case runs
with it. `production` moves to it only if a **live** run passes the gate. An offline
run never promotes: the offline brain ignores prompt wording, so it cannot have
measured the change.

`narrative_judgement.json` is a deliberately weaker prompt. It drops the desk's
written disposition policy in favour of "use your judgement". Run it live and watch
the gate refuse it. The PEP case (ALRT-002) is the one it usually gets wrong.

`critic_source_record.json` is the released Critic prompt: it judges claims against
the source record the investigation gathered. It passed live 20 of 20 and is now the
seed in `src/registry/prompts.py`; see docs/testing-and-evaluation.md for the run.
