# Using Claude Code with This Project

[Claude Code](https://claude.ai/code) is Anthropic's AI coding assistant available as a CLI, VS Code / JetBrains extension, desktop app, and web app. This project ships with custom slash commands, safety hooks, permission rules, and a `CLAUDE.md` context file to make Claude Code immediately productive in this codebase.

This guide covers how to use these features effectively - and how to build up Claude Code's **memory** so it gets better at helping you over time.

---

## Table of Contents

- [Quick Setup](#quick-setup)
- [CLAUDE.md - Project Context](#claudemd--project-context)
- [Skills - Teaching Claude Project-Specific Workflows](#skills--teaching-claude-project-specific-workflows)
- [Hooks - Automated Safety Checks](#hooks--automated-safety-checks)
- [Permissions - Tool Access Control](#permissions--tool-access-control)
- [Memory - Making Claude Code Smarter Over Time](#memory--making-claude-code-smarter-over-time)
- [Subagents - Explore and Plan](#subagents--explore-and-plan)
- [The /loop Command - Recurring Tasks](#the-loop-command--recurring-tasks)
- [Effective Prompting Tips](#effective-prompting-tips)
- [Customizing for Your Workflow](#customizing-for-your-workflow)

---

## Quick Setup

Claude Code works out of the box once installed. No project-specific setup is needed - it reads the configuration files automatically.

1. **Install Claude Code** - follow [official instructions](https://docs.anthropic.com/en/docs/claude-code/overview)
2. **Open this project** in your terminal or IDE
3. Claude Code auto-loads these files on every conversation:
   - `CLAUDE.md` - project context (architecture, commands)
   - `.claude/settings.json` - hooks and permission rules
   - `~/.claude/projects/<hash>/memory/` - your personal memories for this project

---

## CLAUDE.md - Project Context

The `CLAUDE.md` file at the project root gives Claude Code a head start in every conversation. It contains:

- Common commands (how to run tests, start backend/frontend)
- Pipeline architecture (the 5 execution paths, HITL flow)
- Configuration hierarchy (`.env` > country YAML > base YAML)
- Key layers and where they live in the codebase

**You don't need to explain the project structure every time.** Claude Code reads `CLAUDE.md` automatically. Just ask your question directly:

```
# Instead of: "This project uses LangGraph with 7 agents, the graph is in src/agents/graph.py..."
# Just ask:
> Why does the pipeline pause after the fraud crew for claim XYZ?
```

### CLAUDE.md best practices

- **Keep it under 200 lines** - longer files consume context and reduce adherence
- **Be specific** - "Use 2-space indentation" > "Format code properly"
- **Put commands, not procedures** - multi-step workflows belong in skills, not CLAUDE.md
- **Use `@file` imports** for large reference docs: `See @docs/runtime-flow.md for pipeline details`
- **Don't duplicate README** - CLAUDE.md is for instructions to Claude, not project documentation

### Personal overrides with CLAUDE.local.md

Create a `CLAUDE.local.md` in the project root for personal instructions that shouldn't be committed (it's already in `.gitignore`):

```markdown
# My local overrides
- My backend runs on port 9000 instead of 8000
- Use Gemini for my testing (I don't have a Groq key)
- I'm working on the fraud_crew agent this sprint
```

### Subdirectory CLAUDE.md files

You can place a `CLAUDE.md` in any subdirectory. It lazy-loads when Claude reads files in that directory:

```
src/agents/CLAUDE.md       # Loaded when Claude reads agent code
frontend/CLAUDE.md         # Loaded when Claude reads frontend code
```

This keeps context focused - frontend rules don't load when you're working on backend code.

---

## Skills - Teaching Claude Project-Specific Workflows

### What are skills?

Skills are Claude Code's way of packaging reusable workflows as slash commands. Think of them as **playbooks** - instead of explaining the same multi-step task every conversation ("run the tests", "seed the database", "submit a test claim and check the result"), you write the instructions once and invoke them with `/command-name`.

Unlike `CLAUDE.md` (which is always loaded into every conversation), skills are **loaded on demand** - only when you invoke them. This keeps Claude's context focused on what matters right now.

### How this project uses skills

Each skill lives in its own folder under `.claude/skills/` with a `SKILL.md` file:

```
.claude/skills/
  test/SKILL.md              # /test - run pytest with optional filters
  seed/SKILL.md              # /seed - populate test policies
  submit-claim/SKILL.md      # /submit-claim - end-to-end pipeline test
  check-backend/SKILL.md     # /check-backend - verify backend health
  reset-data/SKILL.md        # /reset-data - clean slate for testing
```

Each `SKILL.md` has two parts: **frontmatter** (YAML metadata - name, description, permissions) and **instructions** (markdown that tells Claude what to do when the command is invoked). For example, our `/test` skill tells Claude how to run pytest, how to interpret the `$ARGUMENTS` (keyword filter vs file path), and how to report results.

We set `disable-model-invocation: true` on skills that have side effects (like `/reset-data` and `/submit-claim`) so Claude doesn't accidentally trigger them - only you can invoke those by typing the command.

### This project's slash commands

Type `/` in Claude Code to see available project-specific commands:

| Command | What it does | Example |
|---|---|---|
| `/test [name]` | Run pytest - all, by file, or keyword | `/test pii_masker` |
| `/seed` | Seed US + India test policies | `/seed` |
| `/submit-claim [file]` | Submit a sample claim to running backend | `/submit-claim test_all_paths_india.json` |
| `/check-backend` | Verify backend is running and auth works | `/check-backend` |
| `/reset-data [all]` | Clean data + re-seed. `all` = including users | `/reset-data all` |

### When to use them

- **Starting a dev session**: `/check-backend` then `/seed` to make sure everything is ready
- **After changing agent logic**: `/test` to run the full suite, or `/test fraud` to target fraud-related tests
- **Testing the full pipeline**: `/submit-claim test_all_paths_india.json` to exercise all 5 paths
- **Starting fresh**: `/reset-data` to clean slate without restarting anything

### Creating your own slash commands

Add a new folder under `.claude/skills/`:

```
.claude/skills/my-command/
  SKILL.md          # Required: frontmatter + instructions
  reference.md      # Optional: supporting files referenced from SKILL.md
  scripts/
    helper.sh       # Optional: scripts the skill can run
```

Example `SKILL.md`:

```yaml
---
name: my-command
description: What this command does (shown in autocomplete)
disable-model-invocation: true
allowed-tools: Bash(python *)
argument-hint: "[optional args description]"
---

# Instructions for Claude

Steps to perform when this command is invoked...
```

### All frontmatter fields

| Field | Type | Purpose |
|---|---|---|
| `name` | string | Becomes `/slash-command` |
| `description` | string | When Claude should use it (250 char max) |
| `argument-hint` | string | Autocomplete hint: `[issue-number]` or `[file]` |
| `disable-model-invocation` | bool | `true` = only you can invoke, not Claude |
| `user-invocable` | bool | `false` = only Claude uses it (background knowledge) |
| `allowed-tools` | string/list | Tools Claude can use without approval |
| `model` | string | Override model for this skill (e.g. `claude-opus-4`) |
| `effort` | string | Override effort: `low`, `medium`, `high`, `max` |
| `context` | string | `fork` = run in isolated subagent |
| `agent` | string | Subagent type: `Explore`, `Plan`, `general-purpose` |
| `paths` | string/list | Glob patterns to auto-trigger (e.g. `src/**/*.ts`) |

### Dynamic content in skills

Inject live command output with `!` prefix (runs before Claude sees the prompt):

```markdown
Current git status: !`git status --short`
Active branch: !`git branch --show-current`
```

Use `$ARGUMENTS` for what the user types after the command, or `$0`, `$1` for positional args.

---

## Hooks - Automated Safety Checks

Hooks run automatically at specific lifecycle events. This project has two pre-configured:

### 1. Database Protection (`PreToolUse` on Bash)

Blocks direct deletion of `.db` files and audit logs. If Claude tries `rm data/api.db`, the hook intercepts it and suggests using `python scripts/clean_data.py` instead.

**Why this matters**: Audit logs have a 7-year retention policy for insurance compliance. Accidentally deleting them could be a regulatory issue. The SQLite databases should be cleaned via the script to ensure consistency.

### 2. Environment Check (`UserPromptSubmit`, runs once)

On your first prompt in a session, checks that `.env` exists and has at least one LLM API key. Warns early instead of letting you hit cryptic errors mid-pipeline.

### Adding your own hooks

Edit `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/your-script.sh"
          }
        ]
      }
    ]
  }
}
```

### Hook types

| Type | How it runs | Use case |
|---|---|---|
| `command` | Shell script, reads JSON stdin, controls via exit code | Format, validate, block |
| `prompt` | Single-turn LLM evaluation | Check if action is safe |
| `agent` | Multi-turn subagent with tool access | Complex verification |
| `http` | POST to an endpoint | External integrations |

### Available hook events

| Event | When | Can block? |
|---|---|---|
| `PreToolUse` | Before tool executes | Yes (exit 2) |
| `PostToolUse` | After tool succeeds | No |
| `UserPromptSubmit` | Before Claude processes your prompt | No |
| `SessionStart` | Session begins | No |
| `Stop` | Claude finishes responding | Yes |
| `Notification` | Claude sends a notification | No |
| `SubagentStart` / `SubagentStop` | Subagent lifecycle | No |
| `PreCompact` / `PostCompact` | Context compaction | No |
| `FileChanged` | Watched file changes | No |
| `WorktreeCreate` / `WorktreeRemove` | Git worktree lifecycle | Create: Yes |

Hook exit codes: **0** = proceed, **2** = block (stderr shown to Claude), **other** = proceed with warning.

> **Debug hooks**: Run `/hooks` in Claude Code to see all configured hooks and their matchers.

---

## Permissions - Tool Access Control

Permission rules in `.claude/settings.json` control what Claude can do without asking. This project pre-configures sensible defaults:

**Auto-allowed** (no prompt):
- `pytest`, `python scripts/*`, `uvicorn`, `npm`, `uv pip` commands
- `curl http://localhost:*` (local API testing)
- Reading and searching files

**Blocked**:
- `rm -rf` (destructive deletion)
- Editing `.env` directly (contains secrets)

### Permission rule syntax

```json
{
  "permissions": {
    "allow": ["Bash(pytest *)", "Read"],
    "deny": ["Bash(rm -rf *)"],
    "ask": ["Bash(git push *)"]
  }
}
```

Precedence: **deny > ask > allow**. First matching rule wins.

### Path patterns (gitignore syntax)

```json
{
  "allow": [
    "Edit(/src/**/*.py)",        // Python files in src/
    "Read(configs/**)"           // All config files
  ],
  "deny": [
    "Edit(.env)",                // Protect secrets
    "Edit(data/*.db)"            // Protect databases
  ]
}
```

### Personal overrides

Use `.claude/settings.local.json` (gitignored) for personal permission tweaks:

```json
{
  "permissions": {
    "allow": ["Bash(git push *)"]
  }
}
```

---

## Memory - Making Claude Code Smarter Over Time

Memory lets Claude remember things across conversations - your preferences, project context, decisions, and lessons learned - so you don't repeat yourself.

### How it works

- Memory is stored at `~/.claude/projects/<project-hash>/memory/`
- It's **per-user, per-project** - your memories don't affect other developers
- Claude reads relevant memories at the start of each conversation
- You can ask Claude to remember or forget things at any time
- Memory files are plain markdown - you can edit them directly in `~/.claude/projects/<hash>/memory/`

### What to tell Claude to remember

**About you** (saved once, useful forever):

```
> Remember: I'm the lead developer on this project. I prefer concise responses
  without trailing summaries. I'm experienced with Python but new to React.
```

**Workflow preferences** (saves repeated corrections):

```
> Remember: Always run /test after modifying any agent. Don't amend commits,
  always create new ones. When I say "deploy", I mean push to the main branch.
```

```
> Remember: When I ask about a claim path, show the LangGraph node sequence,
  not the code. I think in terms of the pipeline diagram.
```

**Project context** (decisions and "why"):

```
> Remember: We chose Groq as default provider because the free tier is generous
  enough for demos. Gemini is the fallback for when Groq rate-limits.
```

```
> Remember: The confidence gate thresholds in base.yaml were tuned based on
  testing with 50 sample claims in January 2026. Don't change them without
  re-running the evaluation suite.
```

**External references**:

```
> Remember: Bug tracker is GitHub Issues. The Figma mockups for the HITL
  dashboard are at [URL]. The IRDAI depreciation rules are in docs/irdai-rules.pdf.
```

### What NOT to save to memory

- Code structure, file paths, or architecture (Claude reads these from the code directly)
- Git history or recent changes (Claude uses `git log`)
- Temporary tasks or in-progress work (use Claude's built-in task tracking instead)
- Anything already in `CLAUDE.md`

### Managing memory

```
> What do you remember about this project?     # View current memories
> Remember: [something]                         # Save a new memory
> Forget that I prefer Groq over Gemini         # Remove outdated memory
> Update the memory about thresholds - we       # Update an existing memory
  changed them last week after the March eval
```

Memory files are plain markdown stored in `~/.claude/projects/<hash>/memory/`. You can browse and edit them directly if needed.

### Memory gets better with feedback

When Claude does something you like, say so - it remembers validated approaches:

```
> Perfect, that single PR for the refactor was the right call.
  # Claude saves: "user prefers bundled PRs for refactors in this area"
```

When Claude does something wrong, correct it:

```
> Don't mock the database in integration tests - use the real SQLite.
  # Claude saves: "integration tests must hit real DB, not mocks"
```

These feedback memories prevent the same mistakes across all future conversations.

> **Pro Tip:** Start telling Claude to remember things early - your preferences, project decisions, what worked and what didn't. Memory compounds across conversations and can't be set up in advance. The more you teach it, the less you repeat yourself.

---

## Subagents - Explore and Plan

### Why subagents matter

When you ask Claude Code a question, it works within your **main conversation context** - the same context where you're editing files, running tests, and building features. That context has a finite window, and large search results or deep research can flood it, pushing out the work you care about.

**Subagents solve this.** Each subagent runs in its own **separate, isolated context window**. It does the heavy lifting - searching hundreds of files, reading long outputs, analyzing complex code paths - and returns only a concise summary to your main session. Your main context stays clean and focused.

Think of it like this: instead of you reading through 50 files to answer a question (all those file contents filling your conversation), you send an assistant to do the reading and they come back with a one-paragraph answer.

### Built-in subagent types

Claude Code has three built-in subagent types, two of which are particularly useful for this project:

**Explore agent** - read-only research. Can search files, read code, and fetch web content - but cannot modify anything. Use it when you want Claude to investigate without risk:

```
> Use an Explore agent to find all the places where fraud_score is compared to a threshold
> Use an Explore agent to understand how the memory manager stores episodes
```

**Plan agent** - analysis and architecture. Reads code and designs implementation strategies without making changes. Great for thinking through changes before committing to them:

```
> Use a Plan agent to design how we'd add a new "document verification" agent to the pipeline
> Plan how to add WebSocket support for real-time pipeline status updates
```

**General-purpose agent** - full tool access in isolated context. Can read, write, search, and run commands. Use it for independent tasks that shouldn't interfere with your main session.

### How it works in practice

When Claude decides to use a subagent (or you ask it to), here's what happens:

1. A new agent is spawned with its **own context window** (completely separate from yours)
2. The agent receives a prompt describing what to investigate or plan
3. It works independently - reading files, searching code, analyzing results
4. When done, it returns a **single summary message** back to your main session
5. The subagent's full context (all the files it read, all the searches) is discarded - only the summary remains

This means a subagent can read 30 files and your main context only grows by one message.

### Subagents in skills

You can make a slash command run as a subagent by adding `context: fork` and `agent` to its frontmatter:

```yaml
---
name: investigate
description: Deep-dive into a pipeline issue
context: fork
agent: Explore
---

Investigate $ARGUMENTS thoroughly. Check the graph routing, agent outputs,
HITL triggers, and state transitions. Report findings with file references.
```

When you run `/investigate hitl_checkpoint`, Claude spawns an Explore subagent that digs through the codebase in its own context and reports back without cluttering your main session.

### When to use subagents vs main session

| Use subagent when | Use main session when |
|---|---|
| Research would flood your context with file contents | You need the results immediately for editing |
| You want a fresh, unbiased analysis (no prior assumptions) | Task is simple and direct |
| Exploring unfamiliar parts of the codebase | You know exactly what to change |
| You want to keep your main context clean for implementation | The question is quick to answer |

### Going further: custom agents

This project uses the built-in Explore, Plan, and general-purpose agents - they cover most needs. But if you find yourself repeatedly giving Claude the same specialized instructions (e.g. "audit PII masking for compliance", "review agent prompts for hallucination risks"), you can define **custom agents** in `.claude/agents/`:

```
.claude/agents/
  pii-auditor/
    AGENT.md        # System prompt, tool restrictions, persistent memory config
  prompt-reviewer/
    AGENT.md
```

Each custom agent gets its own system prompt, tool access rules, and optionally its own persistent memory. You'd reference them in skills with `agent: pii-auditor` or invoke them directly.

For example, a PII auditor agent for this project might look like:

**`.claude/agents/pii-auditor/AGENT.md`**:
```yaml
---
name: pii-auditor
description: Audit PII masking coverage across agents and country profiles
allowed-tools: Read Grep Glob
---

# PII Masking Auditor

You are a compliance-focused auditor for the Smart Claims Processor.

## Your job
- Check that all agent prompts in `src/agents/` pass claim data through the
  PII masker (`src/security/pii_masker.py`) before sending to the LLM
- Verify that country profiles in `configs/countries/` define PII patterns for
  all sensitive fields (Aadhaar, PAN, SSN, etc.)
- Flag any code path where raw PII could leak to the LLM or logs

## How to report
- List each file checked with pass/fail
- For failures, show the exact line and what's missing
- End with a summary: "X files checked, Y issues found"
```

You'd then use it from a skill or ask Claude directly:

```
> Run the pii-auditor agent on the current codebase
```

Or wire it into a skill:

**`.claude/skills/audit-pii/SKILL.md`**:
```yaml
---
name: audit-pii
description: Run the PII masking auditor across all agents
context: fork
agent: pii-auditor
disable-model-invocation: true
---

Audit all agent code and country configs for PII masking compliance.
Report findings with file:line references.
```

Now `/audit-pii` spawns the custom agent in its own context, runs the full audit, and returns a summary.

This project doesn't ship with custom agents because the built-in types are sufficient for the workflows here. Consider adding them when you have a recurring, specialized task that needs consistent behaviour across sessions.

---

## The /loop Command - Recurring Tasks

`/loop` runs a prompt or slash command on a recurring interval. Useful for monitoring:

```
/loop 2m /check-backend                           # Check backend every 2 minutes
/loop 5m check if any HITL tickets are pending     # Custom check every 5 minutes
/loop check if tests are passing                   # Claude picks the interval
```

Without an interval, Claude dynamically chooses how often to check (1m to 1h based on what it observes).

### Custom default loop prompt

Create `.claude/loop.md` to define what `/loop` does when invoked without arguments:

```markdown
Check the backend health and report any errors in the last few log lines.
If there are pending HITL tickets, mention them.
```

Then just run `/loop 5m` to repeat that check.

---

## Effective Prompting Tips

### Be direct - Claude already knows the project

Since `CLAUDE.md` provides architecture context, skip the preamble:

```
# Good
> Add a new confidence gate for the communication agent at threshold 0.70

# Unnecessary
> This project uses LangGraph with a StateGraph. There are confidence gates
  defined in base.yaml. I want to add a new one for the communication agent...
```

### Use slash commands for repetitive tasks

```
# Instead of typing the full pytest command each time:
/test hitl_checkpoint

# Instead of explaining how to reset:
/reset-data
```

### Ask for explanations at the right level

```
# Architecture level
> How does the HITL interrupt/resume cycle work end-to-end?

# Code level
> Walk me through route_after_fraud in graph.py - what decides auto-reject vs HITL?

# Debug level
> Claim CLM-12345 got stuck in pending_human_review. What would cause the
  resume to fail after a reviewer approves?
```

### Chain tasks naturally

```
> Add a new agent that checks document expiry dates. It should run after
  intake and before fraud_crew. Add a confidence gate for it. Then run /test
  to make sure nothing broke.
```

### Use /compact for long sessions

If you've been working for a while and Claude starts losing track, run `/compact` to compress the conversation while keeping the important context.

---

## Customizing for Your Workflow

### Settings hierarchy (highest wins)

| Level | File | Scope | Committed? |
|---|---|---|---|
| Managed | System-level `settings.json` | Org-wide | By IT/DevOps |
| Personal | `~/.claude/settings.json` | All your projects | No |
| Project | `.claude/settings.json` | This project, team-shared | Yes |
| Local | `.claude/settings.local.json` | This project, just you | No (gitignored) |

### CLAUDE.md vs Memory vs Skills

| | CLAUDE.md | Memory | Skills |
|---|---|---|---|
| **Scope** | Everyone on the project | Just you | Everyone (or just you) |
| **Content** | Architecture, commands | Preferences, decisions | Procedures, workflows |
| **Updated by** | Manual edits, in git | Conversational | Manual edits, in git |
| **Loaded** | Every session | Every session | On invocation only |
| **Best for** | Facts about the code | Facts about how you work | Multi-step tasks |

**Rule of thumb:**
- Helps every developer? -> `CLAUDE.md`
- Personal preference or non-obvious decision? -> Memory
- Multi-step procedure or workflow? -> Skill (slash command)
- Personal to your machine (ports, keys, paths)? -> `CLAUDE.local.md`

### Useful built-in commands

| Command | What it does |
|---|---|
| `/hooks` | View all configured hooks and matchers |
| `/compact` | Compress conversation to free up context |
| `/loop` | Run a task on a recurring interval |
| `/rewind` | Undo to a previous checkpoint |
| `/help` | Full list of built-in commands |
