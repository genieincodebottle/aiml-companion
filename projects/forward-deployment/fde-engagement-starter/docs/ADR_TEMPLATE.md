# ADR-<NNN>: <short title in the form "we will do X">

> **Why this document exists.** Six months from now, someone at Northwind will
> look at a strange-looking choice in this system and have to decide whether to
> keep it or rip it out. Without a record of what you knew, what you rejected,
> and why, they will guess. An ADR is a message to that person, and they are
> usually not you.
>
> **Who reads it.** The engineer who inherits the system. The security reviewer
> asking why the model runs where it runs. You, in the interview, when someone
> asks you to defend a decision under pressure.
>
> **Time budget: 60 minutes, Day 4. One decision per ADR.** If it takes longer
> than an hour you are either writing a design doc or you have not actually
> decided yet. If you find yourself covering two decisions, split the file.
>
> Save as `docs/ADR-001.md`, `docs/ADR-002.md`, and so on. Numbers never get
> reused, and an ADR is never edited after acceptance - it is superseded by a
> new one that links back.

---

**Status:** <Proposed | Accepted | Superseded by ADR-NNN>
**Date:** <YYYY-MM-DD>
**Deciders:** <names, including who at Northwind agreed>

## Context

<What situation forces a decision now? Facts only. What you observed in the
customer's environment, what constraint applies, what deadline exists. If this
section reads as though the answer is obvious, you have written it after the
fact and stripped out the parts that made it hard.>

## Decision

<One paragraph, in the active voice: "We will <do X>." Not "it was decided" and
not "we considered". State the choice.>

## Alternatives considered

<At least two, and one of them has to be a genuine contender. An ADR whose
alternatives are all obviously bad is an advertisement, not a decision record.>

### Alternative A: <name>
- What it is: <>
- Why we rejected it: <>
- What would make us revisit it: <specific trigger, not "if things change">

### Alternative B: <name>
- What it is: <>
- Why we rejected it: <>
- What would make us revisit it: <>

## Consequences

**What this makes easier:** <>

**What this makes harder:** <every real decision has this section; if yours is
empty, you have not understood the cost yet>

**What it locks in:** <how expensive is reversal in three months, in what units>

## The cost of being wrong

<The section that separates an ADR from a summary. If this decision turns out to
be wrong: how do we find out, how long does that take, what does it cost, and
what is the recovery? A decision that is cheap to reverse deserves less
deliberation than one that is not, and saying which is which is the point.>

- How we would notice: <specific signal, ideally something already measured>
- Time to notice: <>
- Cost if wrong: <>
- Recovery path: <>

## Open questions

<What you decided without knowing. Being explicit here is what lets the next
person tell an informed bet from an oversight.>

---

### What a weak version looks like

A weak ADR is written after the code, lists alternatives nobody seriously
considered, has an empty "what this makes harder" section, and never says how
you would find out you were wrong. It reads as justification rather than
reasoning. The tell is the tone: if it sounds confident all the way through, it
is a press release. Real decisions have a paragraph that is uncomfortable.
