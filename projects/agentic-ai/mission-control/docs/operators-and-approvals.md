# Operators, roles and approvals

A case disposition here is a regulatory filing (a Suspicious Activity Report) or a
decision not to file one. So the platform records who opened every case and who
decided it, and it never lets one person do both.

## Login off (the default)

With `MC_OPERATORS` unset, the console's **Acting as** menu says who you are and sends
it in an `X-Operator` header. Nothing verifies it: anyone who can reach the API can
claim any name. That is fine on your laptop and wrong anywhere else, and the console
says so. The API listens on 127.0.0.1 only, so it is not reachable from other machines
unless you change that.

Two demo people are built in: **ana.analyst** opens cases and **rex.reviewer** decides
them. Open a case as Ana, then switch to Rex to approve it.

## Login on

```bash
MC_OPERATORS="ana:analyst:9f2c...,rex:reviewer:71ab...,lee:analyst+reviewer:c03e..."
```

Each entry is `name:roles:token`. Roles join with `+`. Use long random tokens
(`python -c "import secrets; print(secrets.token_urlsafe(24))"`). Every write then needs
`Authorization: Bearer <token>`, and the console asks for the token instead of a name.

| Action | Needs |
|---|---|
| Open a case (`POST /api/investigations`) | `analyst` |
| Decide a case (`POST /api/decisions`) | `reviewer`, and not the person who opened it |
| Read anything | nothing |

Reads stay open so a dashboard or the event stream works without a token. Put the
whole thing behind your SSO proxy if reads must be private too.

## Maker-checker

The person who opened a case may not approve or reject it, even if they hold both
roles. The runtime enforces this, not the console, so the CLI, the API and any future
client get the same rule. A refused attempt is recorded as a
`maker_checker_blocked` event and the API answers 403.

## The audit trail

Every case records:

| Field | Meaning |
|---|---|
| `started_by` | who opened it |
| `decided_by` | who decided it, or `policy:auto_approve` when `MC_AUTO_APPROVE=1` stood in for a person |
| `decision` | `approve` or `reject` |
| `decided_at` | when (Unix time) |
| `decision_note` | the reviewer's note |
| `dead_letter_reason` | why the case stopped, if it was dead-lettered |

They are on `GET /api/investigations/{id}`, in the console's case panel, in the
`approval_granted` / `approval_rejected` events, and in the checkpoint on disk.

## Decision responses

| Situation | Response |
|---|---|
| Decided | 200 `{"duplicate": false, "status": ...}` |
| The same decision again (a double click, a retry) | 200 `{"duplicate": true}`, nothing happens |
| A different decision on a decided case | 409, naming the first decision and who made it |
| The case is still running, or already finished without a decision | 409 |
| The person who opened the case | 403 (maker-checker) |
| Missing role | 403 |
| No token, or a wrong one (login on) | 401 |
| Unknown case | 404 |
| `decision` is not `approve` or `reject` | 400 |

## After a restart

A case parked at the gate is on disk (`.work/checkpoints/`). After a restart it is
still in **Recent cases**, and deciding it rebuilds it from the checkpoint and event
log without re-running a step. See [ADR 0009](adr/0009-maker-checker-and-durable-decisions.md).
