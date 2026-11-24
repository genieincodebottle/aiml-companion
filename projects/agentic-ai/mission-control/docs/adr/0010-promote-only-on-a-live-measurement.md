# 0010. Promote a prompt only on a live measurement

**Status:** Accepted

## Context

The evaluation printed PROMOTE or BLOCK, but nothing moved: the prompt registry
lived in memory, was rebuilt from source at every start, and the eval never touched a
label. "Eval-gated releases" was a sentence, not a mechanism.

Making the gate move a label raised a second problem. The offline brain decides from
the case data and ignores prompt wording entirely, so an offline run of a candidate
prompt always scores exactly like production. Promoting on that would ship any prompt
at all.

## Options

1. Promote on any passing run.
2. Promote only on a passing live run; an offline run checks the plumbing only.
3. Never promote automatically; print a recommendation.

## Decision

Option 2. The registry is saved to `<data_dir>/prompts.json`. `run_eval --candidate
<file>` registers the new templates under `candidate`, runs every golden case with
them and, if a live run passes, moves `production` and saves. An offline run reports
why it did not promote. Rollback is `cli.py prompts rollback NAME VERSION`.

## Consequences

- A release needs a key and spends real model calls. That is the cost of measuring
  something; the offline run in CI still catches broken plumbing on every push.
- Five golden cases are a smoke test, not a benchmark. A pass means "no regression
  on the cases we have", and live models vary run to run, so use `--repeats`.
- The judge is the same model family as the subject (ADR 0007), so a pass is not an
  independent verdict.
