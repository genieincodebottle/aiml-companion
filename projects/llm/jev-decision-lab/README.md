# Jev Decision Lab

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Model](https://img.shields.io/badge/model-TypeSafe%20Jev-ea580c.svg)
![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)
![Offline](https://img.shields.io/badge/runs%20offline-no%20API%20key-informational.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Blog post:** [Jev brings back the classifier](https://aimlcompanion.ai/blog/jev-system-one-model-classifier-returns-2026). Read it first for what a System One model is, where the idea comes from, and where it breaks down.

A **typed decision model** takes unstructured state and a set of questions, and
returns typed values with probabilities. No text, no JSON to parse, no retry when
the model invents a field. [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
is the first of them.

The interesting question is not whether it is fast. It is **whether the numbers it
returns are good enough to put a threshold on**, and that is a question about your
data, not about the model. This is the harness that answers it.

<details>
<summary><strong>New to the terms?</strong></summary>

- **System One model.** A model that returns decisions instead of text. Named after Kahneman's fast, automatic System 1 thinking.
- **Noul.** Yes or no, returned as a probability between 0 and 1.
- **Choice.** One option from a set, up to 255, with a probability for each.
- **Score.** A position on an ordered scale of 2 to 10 levels. It returns a decimal, the probability-weighted mean of the levels, not a level index.
- **Calibration.** Whether a stated 0.80 really happens about 80% of the time. A confident wrong answer is worse than an unsure one, because a threshold trusts it.
- **ECE.** Expected calibration error, the average gap between what the model promised and what happened. Lower is better.
- **Coverage.** The share of traffic you let the model decide on its own. The rest goes to a person.

</details>

---

## Run it

No API key, no network, no cost.

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/llm/jev-decision-lab
uv sync
```

```bash
uv run python run.py compare       # one holistic question against five narrow ones
uv run python run.py calibration   # what the confidence numbers are actually worth
uv run python run.py threshold     # the band you can automate, and what it covers
uv run python run.py primitives    # one Noul, one Choice and one Score, printed raw
uv run python -m pytest -q         # 25 tests, no network
```

No `uv`? `pip install -e .` and drop the `uv run` prefix. Python 3.10 or newer.

### Against the real model

```bash
pip install typesafe-sdk
```

Then put your key in a file named `.env` in this folder, one line:

```
TYPESAFE_API_KEY=your-key-here
```

```bash
uv run python run.py compare --backend live
```

An environment variable of the same name works too, if you prefer that.

Every command takes `--backend live`. The harness is unchanged, only the source of
the answers moves.

---

## What it measures

`compare` runs the same tickets through two question sets and reports the held-out
half.

| | single | decomposed |
|---|---|---|
| What you ask | **This ticket needs a human specialist now.** | Five narrow questions, one per signal, each answerable from the text alone |
| What decides the outcome | the model's own idea of how much each piece of evidence counts | a logistic regression fitted on **your** labels |
| What you can fix later | a calibration curve | the weights, the calibration curve, and each question separately |

Both arms get one thing fitted on the training half, so neither is handed a free
advantage. Everything reported comes from the half that was held out.

The headline is not accuracy. On the shipped data the two arms land within a
point of each other. What moves is **the share of traffic you can safely
automate**, because decomposed probabilities are better calibrated, so the
confidence threshold that clears your accuracy bar sits lower and lets more
through. Numbers and the exact commands are in [`docs/results.md`](docs/results.md).

### The calibration trap this catches

Temperature scaling is the standard one-parameter fix, and on the single-question
arm it makes things **worse**. Temperature can only make a probability sharper or
flatter. It cannot correct a model that leaned the wrong way, which is what you
get when the model weighed your evidence differently from the way you do. That
needs a shift as well, which is Platt scaling, and the harness reports both so the
difference is visible rather than assumed.

---

## Honest limits

**The shipped tickets are generated, and their labels are computed from the five
signals.** So decomposition wins on this data by construction. That shows the
harness working; it is not a finding about Jev or about any other model. Treat
every offline number as a worked example.

**The simulated backend is not Jev.** It models one mechanism, that a single
holistic question forces a model to guess your weighting while narrow questions do
not, and it reads every narrow signal with the same accuracy in both arms so
nothing is handicapped. It has no relationship to how Jev actually performs.

For a number you can act on, point `--backend live` at 1,000 to 2,000 of your own
decisions that already have verified outcomes. At $0.042 per million input tokens
with output unmetered, that run costs less than lunch, which is the genuinely new
thing here.

---

## Files

| | |
|---|---|
| `src/task.py` | The escalation decision, the five signals, and the generated tickets |
| `src/questions.py` | The two question sets, and the translation to `typesafe_sdk` objects |
| `src/backends.py` | `SimulatedBackend` and `LiveBackend` behind one interface |
| `src/calibrate.py` | Logistic fit, ECE, reliability bins, temperature, Platt, operating point |
| `src/experiment.py` | The two arms, the split, and what gets fitted where |
| `run.py` | The four commands |

`src/calibrate.py` is the reusable part. It has no dependencies and knows nothing
about Jev, so it works on probabilities from any model.

## Licence

MIT. See [LICENSE](LICENSE).
