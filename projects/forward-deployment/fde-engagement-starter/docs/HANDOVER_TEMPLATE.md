# Handover Document - Northwind Freight

> **Why this document exists.** You are leaving. The system stays. This document
> is what determines whether it is still running in six months or quietly turned
> off after the first thing that broke and nobody could fix.
>
> **Who reads it.** The Northwind engineer who inherits this, who was not in any
> of your meetings and does not have your context. Write for them, not for the
> sponsor.
>
> **Time budget: 90 minutes, Day 7, shared with the exec summary. Target 3 pages.**
>
> **The acceptance test is not that you wrote it.** It is that someone at
> Northwind runs the system end to end using only this document, with you in the
> room and silent. Everywhere they get stuck is a gap. Fix the gaps rather than
> answering the question out loud, because next time you will not be there.
>
> Save your filled-in copy as `docs/handover.md`.

---

## 1. What this system does

<Three sentences. What goes in, what comes out, what decision it supports. No
architecture yet.>

**What it explicitly does NOT do:** <the list that prevents someone from
assuming it handles a case it does not. This is the first thing that causes an
incident after handover.>

## 2. Owners

| Component | Northwind owner | Backup | Escalation |
|---|---|---|---|
| Application | <name> | <name> | <name> |
| Evaluation set and gate | <name> | | |
| Legacy API integration | <name> | | |
| Identity and scopes | <name> | | |
| Audit log and retention | <name> | | |

<Every row needs a name. "The platform team" is not an owner, because a team
cannot be paged and cannot be asked a question.>

## 3. How to run it

```
<exact commands, copy-pasteable, for their environment not yours>
```

- Deploy: <>
- Roll back: <exact command, and how long it takes>
- Kill switch: <one command that stops it serving, tested on <date>>
- Logs: <where, and how to read them>
- Healthcheck: <endpoint, what a real failure looks like>

## 4. How to tell if it is working

<Not "is the process up". How do they know the output is still good? Name the
metric, where it is visible, and what number means trouble.>

| Signal | Where | Healthy | Investigate when |
|---|---|---|---|
| <> | <> | <> | <> |

**Re-run the evaluation gate:** `python eval/run_eval.py`. Do this after any
prompt change, model change, or dependency upgrade. A green gate you never saw
go red is not evidence of anything.

## 5. How to change it safely

<The three or four changes they are most likely to want, each with the actual
procedure. Anticipating these is what stops someone editing a prompt in
production on a Friday.>

- **Adding a failure mode:** <steps, including updating golden set and thresholds>
- **Changing a threshold:** <steps, and the requirement to write an ADR>
- **Adding a tool:** <steps, including registering scopes in ROUTE_SCOPES>
- **Rotating credentials:** <steps>

## 6. Known limitations and failure modes

<Be thorough and be honest. A limitation you disclose is a caveat; the same
limitation discovered by a user is a defect, and it costs the system its
credibility.>

| Limitation | Impact | Workaround | Fix if it becomes a problem |
|---|---|---|---|
| <> | <> | <> | <> |

## 7. Decisions and where they are written

| Decision | ADR | Revisit when |
|---|---|---|
| <> | ADR-001 | <trigger> |

## 8. What I would do next

<Ranked, with effort estimates. Your honest engineering opinion, which is worth
more than a roadmap because you have seen the data and they have not yet. Say
which items are worth doing and which are merely possible.>

1. <item> - <effort> - <why it matters>
2. <item> - <effort> - <why>

## 9. Things I did not get to

<And why. Ran out of time, blocked on a decision, judged not worth it. The third
category is the most useful to the person inheriting it, because it saves them
from re-doing your analysis and reaching the same conclusion.>

## 10. Contacts

| Question about | Ask | Until |
|---|---|---|
| <this system> | <you> | <date support ends> |
| <their data> | <name> | ongoing |

---

### What a weak version looks like

A weak handover describes the architecture in detail and says nothing about
operations. It has no "does NOT do" list, no owners with actual names, no kill
switch, and an empty or sanitised limitations section. It is written to look
impressive to the sponsor rather than to be usable by an engineer at 3am. The
test is unchanged: can someone at Northwind run and diagnose this system from
the document alone, while you sit silently and watch?
