# Executive Summary Template

> **Why this document exists.** The exec sponsor funded this and now has to
> decide what happens next. They will give it about four minutes, on a phone,
> between meetings. This document has one job: make the decision available to
> them without requiring them to reconstruct it from the method.
>
> **Who reads it.** The exec sponsor, and whoever they forward it to. Not
> engineers. The engineering audience is served by `docs/handover.md`.
>
> **Time budget: 90 minutes, Day 7, shared with the handover doc.**
>
> **HARD CONSTRAINT: ONE PAGE. Roughly 500 words.** Not "one page plus an
> appendix". Not "one page at 9pt". If it does not fit, the thinking is not
> finished - a second page is nearly always the method leaking in.
>
> **LEAD WITH THE DECISION, NOT THE METHOD.** The first line the sponsor reads
> must be what you are recommending. How you built it, what the architecture is,
> which model you chose, and how the retrieval works are all irrelevant to the
> decision and belong in the handover. Every sentence in this document should
> survive the question "does this change what they decide?"
>
> Save your filled-in copy as `docs/exec-summary.md`.

---

# <Engagement name> - Executive Summary

**To:** <sponsor name> | **From:** <your name> | **Date:** <YYYY-MM-DD>

## Recommendation

<One or two sentences. What you recommend they do next: proceed to production,
extend the pilot, stop, or pivot. State it as a recommendation, not as a menu.>

## Why

<Three bullets maximum. Business reasons, expressed in Northwind's units -
deliveries, hours of dispatcher time, cost per failed delivery. Not accuracy
percentages unless the sponsor already thinks in them.>

- <>
- <>
- <>

## What we found

<Two or three sentences on the state of the world that they did not know before
this engagement. This is often the most valuable thing you deliver, and it is
frequently not the software. Example shape: which failure mode actually drives
the cost, or a process assumption that turned out to be false.>

## What it costs to proceed

| | |
|---|---|
| Time to production | <> |
| Run cost | <per month, at their volume> |
| What Northwind must provide | <people, access, decisions> |

## Risks

<Two, at most three. The ones that could change the decision. Each one sentence,
with the mitigation in the same sentence.>

- <risk, and what reduces it>
- <risk, and what reduces it>

## What we need from you

<Specific, and by when. A summary that ends without an ask ends in nothing
happening.>

- <decision or resource> by <date>

---

### What a weak version looks like

A weak exec summary opens with "Over the past seven days we ingested the
operational exports and built a hybrid retrieval pipeline...". That is the
method, and by the time the sponsor reaches the recommendation they have already
skipped to the end or stopped reading. Other tells: percentages with no
business unit attached, no cost, no ask, no risks (which reads as either naive
or evasive), and a second page. If your reader has to scroll to find out what
you want them to do, rewrite it from the recommendation up.
