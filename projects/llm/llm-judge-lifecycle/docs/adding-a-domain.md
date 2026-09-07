# Adding a domain

Adding a domain costs **data and prose**. It costs no Python.
`tests/test_domain.py::test_no_domain_specific_logic_in_src` fails the build if a
domain name is ever hardcoded in `src/`, so this is checked rather than promised.

Work through it in this order. Step 2 is the one that takes real time, and
skipping it is how people end up with a judge that is carefully aligned to
nothing.

---

## 1. Three files

```
domains/<your-domain>/
├── domain.yaml      criteria, guidelines, generation instructions
├── records.json     the subjects, and everything an artefact may claim
└── labels.jsonl     human-labelled examples with rationales
```

Then one line in `configs/base.yaml`:

```yaml
domain: <your-domain>
```

## 2. Write the criteria

Three or four is right. More than that and raters stop agreeing with each other,
which widens the drift band until it detects nothing.

**Split must-have from soft, and be strict about it.** A must-have failure drops
the artefact; a soft failure is recorded and served anyway. Collapse the two and
you get either a gate that rejects accurate work over a style preference, or a
gate that ships falsehoods. Teams that discover they need the distinction usually
discover it from the first of those.

Each criterion needs a `guideline`, and it does double duty: it is what human
raters label against **and** the seed rubric RART starts from. One field, two
consumers, on purpose. Teams that write them separately find they drift apart
within a month - the rater guidance gets a clarification, the judge prompt does
not, and judge-human disagreement is now measuring a documentation gap. You will
spend a week tuning the judge before anyone notices.

A guideline that works has four parts:

```yaml
- id: supported
  display: Backed by the policy record
  must_have: true
  failure_modes: [unsupported_claim, misattributed_detail]
  guideline: |
    PASS when ...                  # the pass condition
    FAIL when ...                  # the fail condition
    FAIL ALSO when ...             # the near-miss you keep seeing
    BOUNDARY: "..." is a PASS.     # at least one case close to the line
    "..." is a FAIL.

    [grounded]                     # optional machine-readable tags
```

Write the BOUNDARY case last and make it genuinely hard. It is the part raters
actually use, and a guideline without one produces labels that disagree.

### Tags, if you want the offline baseline

| tag | effect | failure mode |
|---|---|---|
| `[grounded]` | numbers and named entities must appear in the record | `unsupported_claim` |
| `[require: subject.field]` | that field's value must appear in the artefact | `missing_subject` |
| `[banned: "a", "b"]` | substring rejection | `generic_filler` |
| `[spoiler: "a", "b"]` | substring rejection | `spoiler` |
| `[sensitive: "a"]` | substring rejection | `sensitive_framing` |
| `[min_words: n]` / `[max_words: n]` | length floor / ceiling | `too_short` / `too_long` |

**Double quotes only.** A single quote is not a delimiter here, because the
phrases worth banning contain apostrophes. `src/rules.py` documents what happened
when it was.

Tags are optional. Without them the criterion still works against a model judge;
you just lose the free baseline and the ability to run Phase II offline.

## 3. Write the records

```json
{
  "id": "tkt-001",
  "added_week": 0,
  "subject":    { "reference": "ORD-48812", "issue": "..." },
  "references": [{ "title": "Missing-item policy", "attributes": ["..."] }],
  "facts":      ["Replacement dispatch takes 2 working days."],
  "context":    { "tier": "standard" }
}
```

`subject`, `references` and `facts` together are the **closed set of things an
artefact may assert**. If a claim is not derivable from them it is ungrounded, so
be deliberate: anything you leave out becomes a hallucination when the generator
writes it, and anything you put in becomes fair game.

Set `added_week` on later arrivals. Phase IV's shift check keys on it, and
without any recently-added records that check cannot run - which the monitor
reports as a gap rather than a pass.

## 4. Label examples - the expensive step

Aim for **20+ per must-have criterion**, held near 50/50 PASS/FAIL.

Balance is not optional. Real defect rates are a few percent, so a naturally
sampled set is ~95% PASS and a judge answering PASS to everything scores 95%
while catching nothing. Balancing makes specificity measurable; the cost is that
the numbers no longer estimate a live defect rate, and you must say so wherever
you report them.

```jsonl
{"id": "sup-01", "record_id": "tkt-001", "artefact": "...",
 "labels": {"supported": "FAIL"},
 "rationales": {"supported": "The record says 2 working days. 24 hours is invented."},
 "failure_modes": {"supported": "unsupported_claim"},
 "source": "expert"}
```

**Every FAIL needs a rationale.** The loader refuses to start without one, and
the reason is worth understanding: the reasoning meta-judge compares the judge's
stated reason against the human's, so an example with no rationale is invisible
to reasoning agreement. Your benchmark keeps its row count while the signal RART
depends on silently shrinks. Ten extra seconds per rater is the difference
between a judge that is right and a judge that is right for the right reason.

Include cases your rules **cannot** catch. They are the headroom the model judge
has to earn, and without them you cannot tell whether an LLM judge is worth its
latency and its bill. Both shipped domains mark these `BASELINE-MISS`.

## 5. Run it

```bash
python run.py --offline benchmark                      # composition, splits, warnings
python run.py --offline eval --criterion <id> --split validation
python run.py --offline tune --criterion <id>
python -m pytest tests/test_domain.py -q                # validates your data
```

The benchmark report warns about thin splits. Believe the warnings: a validation
split with two FAIL examples measures specificity to the nearest fifty points.

## 6. Weekly review data, when you have it

```
domains/<your-domain>/hitl/week_NN.jsonl
```

```jsonl
{"id": "hitl-w6-01", "record_id": "tkt-001", "criterion": "supported",
 "artefact": "...", "outcome": "served_without_revision",
 "rater_labels": ["FAIL", "FAIL", "PASS"], "judge_label": "PASS",
 "rationale": "...", "failure_mode": "unsupported_claim"}
```

An **odd** number of raters, at least three. With an even panel a tie has to
break somewhere, and any consistent tie-break silently biases the whole week's
measurement in that direction.

---

## What ports, and what does not

The **mechanism** ports completely: the gate that drops, the critic whose reason
drives the next draft, the band pegged to rater disagreement, the retry curve,
the append-stable splits.

The **criteria** do not, and should not. A spoiler is meaningless in customer
support; an unauthorised refund commitment is meaningless in a film catalogue.
The two shipped domains share exactly one criterion id (`concise`, a length
check) and nothing else. A team reusing another team's rubrics is porting the
wrong half.

What *does* transfer between domains is the **shape of the failures**. Both
shipped domains contain a real number attached to the wrong noun, a claim that
contradicts the record rather than adding to it, and a polite well-formed
sentence that says nothing. Those three will be in your domain too. Write them
into your benchmark before you go looking for them in production.
