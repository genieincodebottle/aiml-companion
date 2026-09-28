# Hands-On Exercises

Graded challenges. Difficulty runs from warm-up to stretch. Everything except #6 and #7
runs offline with zero credentials. Read the [Learning Guide](LEARNING_GUIDE.md) first.

**Windows PowerShell:** the `VAR=value command` lines below are POSIX shell syntax
(see the README's Windows note) - in PowerShell, set `$env:VAR = "value"` first, then
run the command on its own line.

---

## 1. Read a trajectory like an operator (warm-up)

Run the fleet and follow one repo's event stream end to end:

```bash
cd backend
uv run python cli.py stream datetime-fleet
```

**Your task:** in the output for `auth-service`, point at (a) the retrieval call, (b) the
cross-repo memory recall, (c) the guardrail block, (d) the loop iterating on a real test
failure, and (e) the review verdict. Annotated answer key: Learning Guide §3.

**What you learn:** event streams are how you debug agents in production. If you cannot
read a trajectory, you cannot operate an agent.

---

## 2. Write a migration rule end to end (the core exercise) (moderate)

The fixture repo [`metrics-service`](../backend/fixtures/repos/metrics-service/) uses the
deprecated `datetime.utcfromtimestamp()` - and the rulebook has **no rule for it**. Ship
one.

**Your task:**
1. In [`src/rulebook/rules.py`](../backend/src/rulebook/rules.py), add a
   `modernize-utcfromtimestamp` rule: `guidance` for the live worker, `detect` regex,
   and `edit_steps` for the stub worker (mirror `modernize-datetime`).
2. Add a job `epoch-fleet` targeting `metrics-service` to `_STATIC_JOBS`.
3. Run it and prove it worked:
   ```bash
   uv run python cli.py run epoch-fleet
   git -C .work/remotes/metrics-service.git log --oneline --all   # your commit, on a real branch
   ```

**Success criteria:** tests pass, 1 file changed, `tampered False`, outcome `pr_open`,
and the migration branch exists on the bare remote.

**Watch out for:** `window_start` wraps its argument in parentheses -
`utcfromtimestamp(ts - (ts % seconds))`. A non-greedy `(.*?)\)` capture will cut the
argument at the *inner* `)` and produce broken code that the tests catch. That failure
is worth experiencing once before you read the solution.

<details>
<summary><b>Solution</b> (write yours first)</summary>

```python
"modernize-utcfromtimestamp": MigrationRule(
    id="modernize-utcfromtimestamp",
    name="Modernize deprecated datetime.utcfromtimestamp()",
    summary=(
        "Replace the deprecated, naive datetime.utcfromtimestamp(x) with the "
        "timezone-aware datetime.fromtimestamp(x, timezone.utc)."
    ),
    guidance=(
        "Python 3.12 deprecated datetime.utcfromtimestamp(): it returns a NAIVE "
        "datetime. Replace every call `datetime.utcfromtimestamp(X)` with "
        "`datetime.fromtimestamp(X, timezone.utc)` and ensure `timezone` is imported "
        "from the datetime module. Do not touch the tests; run them until green."
    ),
    detect=r"datetime\.utcfromtimestamp\(",
    tags=("python", "datetime", "epoch", "deprecation", "timezone", "py312"),
    edit_steps=(
        EditStep(
            description="Replace utcfromtimestamp(x) with fromtimestamp(x, timezone.utc)",
            kind="regex_replace",
            # Greedy capture: the call is the last `)` on its line, so `.*` correctly
            # spans nested parens like `ts - (ts % seconds)`.
            find=r"datetime\.utcfromtimestamp\((.*)\)",
            replace=r"datetime.fromtimestamp(\1, timezone.utc)",
        ),
        EditStep(
            description="Ensure `timezone` is imported from datetime",
            kind="ensure_import",
            module="datetime",
            symbol="timezone",
        ),
    ),
),
```

And the job:

```python
"epoch-fleet": MigrationJob(
    id="epoch-fleet",
    title="Modernize datetime.utcfromtimestamp() in metrics-service",
    rule_id="modernize-utcfromtimestamp",
    targets=(RepoTarget(kind="fixture", ref="metrics-service", name="metrics-service"),),
    description="Exercise #2: a learner-authored rule applied through the full pipeline.",
),
```

**Stretch:** add `metrics-service` expectations to `evaluation/golden.py` and extend
`run_eval.py` to run both jobs, so your rule is regression-guarded forever.
</details>

**What you learn:** the anatomy of a rule (LLM guidance vs scripted steps), and that the
platform is *extensible by data* - adding a capability touched no orchestration code.

---

## 3. The defense-in-depth lab (moderate)

An agent can "pass" the tests by editing them. Watch each safety layer catch it.

```bash
# Layer 1 ON (normal): the PreToolUse hook blocks test-file writes.
uv run python cli.py stream datetime-fleet | grep guardrail_block

# Layer 1 OFF (teaching flag): the worker now silently rewrites test files...
ME_DISABLE_GUARDRAILS=1 uv run python cli.py stream datetime-fleet | grep -E "review|repo_done"
# ...tests are still green (the trap!) - but the REVIEWER flags tampered=true
# and every repo ends 'escalated' instead of 'pr_open'.
```

Then read the executable proof: [`tests/test_defense_in_depth.py`](../backend/tests/test_defense_in_depth.py).

**Questions to answer:**
1. With the hook off, why did the tests still pass even though test files were edited?
   (Hint: what exactly did the regex sweep change in them?)
2. If the reviewer were ALSO disabled, what third layer still catches this before a
   deploy? (Look at `no_tamper_rate` in [`evaluation/scorers.py`](../backend/evaluation/scorers.py).)
3. Why must the tamper check be *deterministic code reading git*, rather than asking an
   LLM "did you modify the tests?"

**What you learn:** never rely on one control. Outcome checks (tests green) can be
gamed; process checks (what did git actually change) are much harder to fool.

---

## 4. Budget squeeze (moderate)

```bash
ME_MAX_AGENT_STEPS=3 uv run python cli.py run datetime-fleet
ME_MAX_AGENT_STEPS=3 uv run python -m evaluation.run_eval; echo "exit=$?"
```

**Your task:** explain which repos escalate and why (`billing-service` needs fewer steps
than the others - see their READMEs), and why the eval's non-zero exit matters for CI.

**What you learn:** budgets are the difference between "escalate to a human" and "burn
money forever". Non-termination is the #1 failure mode of naive agents.

---

## 5. Add a guardrail (challenging)

The hook blocks test edits, path traversal, and bad extensions - but nothing stops the
worker writing to CI config (imagine it "fixing" a failing pipeline by editing
`.github/workflows/*.yml`).

**Your task:** in [`src/guardrails/hooks.py`](../backend/src/guardrails/hooks.py), block
writes to `.github/`, `*.yml`/`*.yaml`, and dotfiles. Add tests in
`tests/test_guardrails.py` proving both the block and that normal source writes still
pass. Then ask: should this be a *deny* or a *require-human-approval*? Defend the choice.

**What you learn:** guardrail design is least-privilege thinking - and the deny/approve
distinction is a real production decision, not a technicality.

---

## 6. Upgrade the reviewer to an LLM critic (needs an API key) (challenging)

The reviewer is deterministic. Augment it: in `live-sdk` mode, ALSO send the diff to
Claude for a qualitative critique (minimality, style, missed occurrences) and attach the
prose to the verdict's `reasons`.

**Hard requirement:** the deterministic tamper gate must remain and must veto. Write one
sentence for the PR description explaining why the LLM's opinion is advisory but the git
check is binding.

**What you learn:** the LLM-as-judge pattern, and where its authority must stop.

---

## 7. Run it against a real GitHub repo (challenging)

Create a throwaway repo with one `datetime.utcnow()` call and a test, then follow the
"Run against your REAL GitHub repos" section of the [README](../README.md). Approve the
PR in the console and open the PR link on GitHub.

**What you learn:** the offline demo and the real thing are the same code path - which
is the entire point of the provider abstraction in [`src/vcs/`](../backend/src/vcs/).

---

## 8. Stretch goals (pick one)

- **LangGraph interrupt:** replace the batch path's `approval_policy` with a real
  `interrupt()` checkpoint + checkpointer, so a batch run can pause for approval and
  resume - durable human-in-the-loop.
- **Concurrency study:** instrument `ME_FAN_OUT` from 1 to 4 and chart wall-clock vs
  cost. Where does the semaphore bind, and why is the approval gate deliberately
  *outside* it? (Read `_run_repo` in `engine.py` before answering.)
- **New tool:** add a `pip_check` tool (runs `pip check` in the worktree) through
  `ToolSpec` -> registry -> both workers, and watch it surface in the live SDK worker's
  MCP server with zero changes to `sdk_agent.py`.

**What you learn:** each of these is a real production backlog item; doing one gives you
a genuine "then I extended it by..." interview story.
