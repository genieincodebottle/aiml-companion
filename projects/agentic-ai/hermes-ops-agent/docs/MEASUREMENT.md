# What this project measures, and why it is easy to get wrong

Hermes claims a learning loop: once it has written a skill for a task, doing
that task again costs less. This project checks that claim on your own agent
instead of taking it on trust.

The measurement is harder than it looks, and the reason is worth understanding
before you read any number the tool prints.

## The confound

A second run of the same task is cheaper for two independent reasons.

**The skill made it shorter.** Fewer tool calls, fewer trips round the agent
loop, because the procedure was written down instead of rediscovered. This is
the loop working.

**The prompt prefix was cached.** Hermes builds the prompt stable-first
(identity, then context files, then skill metadata, then the memory snapshot,
then the conversation) precisely so providers can cache a prefix of it. Cached
input tokens are billed at a fraction of normal input tokens, or not at all.
This has nothing to do with the skill.

A total-token delta mixes these. On a warm cache it will show a large
"improvement" on a run where the skill was never loaded.

## What the tool reports instead

| Metric | Why it is there |
|---|---|
| `tool_calls` | The headline. Caching cannot remove a tool call. If the skill shortened the procedure, this drops. |
| `api_calls` | Iterations of the agent loop. Same reasoning as `tool_calls`. |
| `uncached_input` | `input_tokens - cache_read_tokens`, floored at zero. |
| `output_tokens` | Never cached. Always real work. |
| `billable_work` | `uncached_input + output_tokens`. Closest single number to what it cost, without needing prices. |
| `cache_read_tokens` | Shown so you can see the confound rather than have it hidden. Higher is better here, which is why it is the one metric flagged as such. |
| `total_tokens` | Included only so you can see how misleading it is next to the others. |

The verdict line is deliberately conservative. Fewer tool calls credits the
skill. The same tool calls with fewer tokens says "likely caching, not the
skill". More tool calls says the skill probably did not match, and points you
at reading it.

## The caveat the tool cannot resolve for you

Whether `input_tokens` already excludes cache reads is provider-dependent.
Anthropic reports them as separate fields. Some OpenAI-compatible gateways fold
cache reads into the input count.

The tool cannot tell which one you are on, so `uncached_input` is floored at
zero and every raw field is printed alongside the derived one. If
`cache_read_tokens` is larger than `input_tokens` on your provider, the fields
are separate and `uncached_input` is an undercount. That is visible in the
output rather than buried.

## Where the numbers come from

Everything is read from Hermes' own SQLite store, by default
`~/.hermes/state.db`, honouring `HERMES_HOME` so profiles work.

The `sessions` table carries per-session `tool_call_count`, `api_call_count`,
`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`,
`reasoning_tokens` and `estimated_cost_usd`. The `messages` table carries one
row per turn with `role`, `tool_name` and `tool_calls`.

Two consequences:

- **This works for any session**, from the CLI, the gateway or cron. It is not
  limited to one-shot runs.
- **It is provider-independent.** Whatever model produced the session, the
  accounting is the same.

`sessions.tool_call_count` is the number to trust. The tool also recounts tool
rows from `messages` as a cross-check, and warns when the two disagree, which
usually means the session ended abnormally.

## The database is opened read-only

`state_db.py` connects with `file:...?mode=ro`. The file being read is your
live agent memory, and SQLite will let an analysis tool corrupt it if opened
read-write. A test asserts the connection refuses writes and that the file
bytes are unchanged after a full read.

## Running a fair comparison

The comparison is only meaningful if the two runs differ in one thing.

1. Same task, worded the same way.
2. Same model. The tool warns when they differ.
3. Both fresh sessions. A continuation inherits context the baseline never had,
   so the tool warns when `parent_session_id` is set.
4. The task has to actually do something. Two sessions with zero tool calls
   have no procedure to shorten, and the tool warns about that too.

If a warning fires, read it before the table. Warnings print above the numbers
on purpose: a caveat underneath a table gets skipped.
