# Scoping Document - Northwind Freight

> **Why this document exists.** It converts "they want to reduce failed
> deliveries using AI" into something that can be built and, more importantly,
> into something that can be declared finished. Without it, the engagement has
> no edge, and an engagement with no edge does not end, it just runs out of
> budget.
>
> **Who reads it.** The Northwind exec sponsor, who is checking that you heard
> the business problem. The ops lead, whose team has to use the result. Your own
> delivery lead, who is checking that you did not agree to something impossible.
>
> **Time budget: 90 minutes, Day 1. No code. No edits after the timer fires.**
> The constraint is the exercise. A scoping doc you keep revising while you build
> is a diary, not a scope, and it cannot be used to say no to anything.
>
> Save your filled-in copy as `docs/scoping-doc.md`.

---

## 1. The business problem, in the customer's words

<Write the problem as the ops lead would say it out loud, not as a technical
statement. If you cannot write it without using the word "AI", you have not
found the business problem yet.>

<What does a failed delivery cost Northwind today? If you do not know, that is
your first question to them, and note here that it is unanswered.>

## 2. What "better" means

<One measurable statement. Not "improve triage accuracy" - something like "the
dispatch desk stops re-reading tickets that only need an address correction".>

| Metric | Today | Target | Who measures it | How they measure it |
|---|---|---|---|---|
| <metric> | <baseline, or UNKNOWN> | <target> | <name> | <method> |

<If the "Today" column is UNKNOWN, say so plainly. An improvement measured
against a baseline nobody recorded is not an improvement, it is an assertion.>

## 3. In scope

<Bounded list. Each item small enough that you can say whether it is done.>

- <item>
- <item>

## 4. Explicitly OUT of scope

<This section is the one that earns the document. Name the things they will ask
for in week two, and rule them out now while nobody is disappointed yet.>

- <item, and one clause on why not now>
- <item, and one clause on why not now>

## 5. Success criteria

<How you and Northwind will both know this is done. These should map onto
`make check` and onto the eval gate, so that "done" is observable rather than
negotiated at the end.>

- [ ] <criterion>
- [ ] <criterion>

## 6. Constraints

| Constraint | Detail | Source |
|---|---|---|
| Data residency | <what cannot leave, and whose rule it is> | <name / policy> |
| Identity | <their IdP, their groups> | <name> |
| Legacy systems | <what you must integrate with as-is> | <name> |
| Timeline | <dates, and what is fixed vs preferred> | <name> |
| Budget | <cost ceiling, per what unit> | <name> |

## 7. Assumptions

<Everything you are proceeding on without confirmation. Each one gets an owner
and a date by which it must be confirmed or it becomes a risk.>

| Assumption | If wrong, what breaks | Owner | Confirm by |
|---|---|---|---|
| <assumption> | <impact> | <name> | <date> |

## 8. Open questions

<Questions you could not get answered in the 90 minutes. Listing them is not a
failure; pretending you had answers is.>

1. <question> - blocking / non-blocking - ask <name>

## 9. Rough shape of the solution

<Three to five sentences. Not an architecture. Enough that the sponsor can
picture it and object if it is not what they imagined.>

## 10. What could make this fail

<Not a risk register. The two or three things that genuinely sink engagements
like this one. Be specific to Northwind.>

- <risk> - <what you will do about it in week one>

---

### What a weak version looks like

A weak scoping doc has an empty or generic "out of scope" section, targets that
are directional rather than measurable ("improve efficiency"), no named owners,
and no open questions - which really means the questions were never asked. It
reads as though everything is agreed. Six weeks later the customer says "we
assumed it would also do X", and there is no document that says otherwise.
