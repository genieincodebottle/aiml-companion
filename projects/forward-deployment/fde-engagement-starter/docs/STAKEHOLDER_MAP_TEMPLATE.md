# Stakeholder Map - Northwind Freight

> **Why this document exists.** Engagements fail on people far more often than
> on technology. The system that never shipped was usually blocked by someone
> nobody had spoken to, and the system that shipped and was never used was
> usually built for someone who did not have to use it.
>
> **Who reads it.** Mostly you. Occasionally your delivery lead, when something
> is stuck and you need to explain where. Do not circulate it at Northwind: it
> contains candid assessments, and a stakeholder map written for an audience is
> a stakeholder map that is no longer true.
>
> **Time budget: 120 minutes, Day 2.** Longer than Day 1 on purpose, because
> most of the time goes on conversations rather than on writing.
>
> Save your filled-in copy as `docs/stakeholder-map.md`.

---

## 1. The map

| Name | Role | Signs off? | Can block? | Has to use it? | Current stance | Last contact |
|---|---|---|---|---|---|---|
| <name> | <exec sponsor> | yes | yes | no | <champion / neutral / sceptical> | <date> |
| <name> | <ops lead> | no | yes | yes | <> | <date> |
| <name> | <platform / security> | no | yes | no | <> | <date> |
| <name> | <dispatcher, daily user> | no | no | yes | <> | <date> |
| <name> | <data owner> | no | yes | no | <> | <date> |

Three columns matter most and they are rarely the same person:

- **Signs off** funds it and declares it successful.
- **Can block** can stop it without needing anyone's permission. Security and
  data owners usually can. They are often not in the kickoff meeting.
- **Has to use it** decides whether it is still running in six months.

## 2. Who has not been spoken to yet

<List them. This is the section that stops an engagement from being blindsided.
Anyone in the "can block" column you have not met is a live risk, today.>

- <name> - <why they matter> - reaching out by <date>

## 3. The person whose job this changes

<Name them. Every automation changes someone's day, and that person's opinion
outranks the sponsor's enthusiasm on the question of whether it gets used.>

- Name: <name>
- What their day looks like now: <description>
- What it looks like after: <description>
- What they gain: <specific>
- What they lose: <be honest - status, autonomy, a task they enjoyed, headcount>
- Have you asked them directly? <yes / no, and what they said>

## 4. Conflicting incentives

<Where two stakeholders want incompatible things. Do not smooth this over.
Naming the conflict early is what lets the sponsor resolve it while it is still
cheap.>

| Stakeholder A wants | Stakeholder B wants | Conflict | Who resolves it |
|---|---|---|---|
| <> | <> | <> | <name> |

## 5. Communication plan

| Who | What they need to hear | How often | Format |
|---|---|---|---|
| <sponsor> | progress against the outcome, risks | <weekly> | <5-line email> |
| <ops lead> | what changes for the team, when | <twice weekly> | <standup> |
| <security> | anything touching data or identity | <on change> | <written> |

## 6. Escalation path

<In order. Who you go to first, second, third, and the trigger for each step. If
you have to invent this during an incident, you will invent it badly.>

1. <name> - for <what> - trigger: <what has to be true>
2. <name> - for <what> - trigger: <>

---

### What a weak version looks like

A weak stakeholder map is the org chart with a column added. It lists titles
instead of stances, has nobody in the "not spoken to yet" section, records no
conflicts, and never names the person whose job changes. It is comfortable to
write and predicts nothing. The test: does this document tell you who to call
when the project stalls next Tuesday? If not, it is decoration.
