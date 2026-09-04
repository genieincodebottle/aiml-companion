# Credit Risk Pipeline

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

An end-to-end credit default pipeline on the German Credit data: cleaning, EDA,
feature engineering, cross-validated training, cost-sensitive threshold tuning,
SHAP adverse-action reasons, a FastAPI endpoint and PSI drift monitoring.

The interesting part is not the model. Two differently-shaped models tie at
**0.78 AUC**, and that is roughly where this dataset tops out. The interesting
part is everything that decides whether a 0.78 model is safe to deploy: which
rows it was scored on, which column it was not allowed to see, and which
threshold it ships with.

**New here? Start with [section 8](#8-the-notebook).** The notebook needs no
install.

## 1. The problem

A bank scores loan applicants. Approving someone who defaults costs roughly ten
times what rejecting someone who would have repaid costs, so accuracy is the
wrong objective and 0.5 is the wrong threshold.

That much is standard. What makes credit risk different from an ordinary
imbalanced classification problem is that the decision is *regulated*. You have
to be able to tell a rejected applicant why, in writing, which rules out a model
you cannot explain. And you are forbidden from using certain columns at all,
however predictive they are.

So there are several things to get right at once, and this project got each of
them wrong at some point in its history. Sections 2 to 6 are those mistakes,
kept because the mistake is the lesson.

## 2. Why the reported cost was almost zero

Evaluation called `pipeline.predict_proba(X)` on a model fitted to all of `X`. A
gradient booster memorises 1,000 rows without effort, so it scored its own
training data almost perfectly and the pipeline reported a **total
misclassification cost of $6**, against a real $662.

The flattering number was not the expensive part. The threshold was chosen from
those memorised scores, and *that* threshold is what got written to
`threshold.json` and loaded by the serving API:

| | threshold | reported cost | real cost |
|---|---|---|---|
| in-sample (old) | 0.31 | **$6** | **$1,189** |
| out-of-fold (now) | 0.01 | $662 | $662 |

Because false negatives cost 10x false positives, the honest threshold is
*aggressive*. The overfitted model looked so confident that 0.31 seemed safe,
and shipping it costs **$1,189 against $662** - about 1.8x a correctly tuned
threshold.

Everything downstream now runs on `cross_val_predict` output: every row scored
by a copy of the model that never trained on it. `full_evaluation` raises rather
than falling back to in-sample scoring when those out-of-fold predictions are
missing, because a quiet fallback is how this happened the first time.

## 3. The threshold is a corner, and the pipeline says so

Read that table again. The tuned threshold is **0.01**, and at 0.01 the model
flags **91% of all applicants** and rejects **612 of the 700 good borrowers**.

It is the correct answer to the question as posed, and it is not a lending
policy. A 10:1 cost ratio with no constraint on approval rate has an obvious
degenerate solution: reject almost everyone, because a false positive is cheap
and you are being scored only on cost. Nothing in the objective says the bank
also has to write loans.

This is the trap the project now refuses to walk into quietly.

- The search grid used to start at **0.05**, and it returned exactly 0.05 - the
  first value it was allowed to try. An argmin sitting on the boundary of its
  own grid has not found a minimum. The bound starts at 0.01 now, and the true
  argmin turns out to live there: $662, against the $685 the walled-off search
  reported.
- `_warn_if_degenerate()` fires on every run: once when the optimum lands on the
  grid edge, once when the chosen threshold flags more than half of all
  applicants, naming how many good borrowers that rejects.

The number still gets written to `threshold.json` and handed to the API, because
suppressing it would be its own kind of dishonesty. But it goes out labelled.

**A cost-minimising threshold is not automatically a shippable one, and the way
you find out is by printing what it does to the approval rate.**

## 4. The column that had to go

`personal_status` reads "male single", "female div/dep/mar". It encodes **sex**,
and it was going into the model as an ordinary categorical. ECOA / Regulation B
prohibits that outright.

The usual objection is that accuracy requires it. It does not:

| Logistic Regression | cross-validated AUC |
|---|---|
| with `personal_status` | 0.7886 |
| without it | 0.7828 |

Six thousandths of AUC, against a cross-validation standard deviation of 0.025.
What it buys on the other side of the ledger: the female
default rate in this data is **35.2%** against 27.7% for men, so a model holding
that column learns to charge women more for being women - 31% of applicants.

`drop_protected` is on by default. Dropping the column is the *floor*, not the
finish line: proxies survive, which is why a real deployment needs a
disparate-impact audit rather than a deleted column.

## 5. What this dataset cannot support

This README used to advertise DTI ratio and credit utilization as domain
features. Neither can exist here.

- **German Credit has no income column.** DTI resolved `income` to
  `personal_status` - a categorical - `to_numeric` coerced every value to
  `NaN`, and the result was a **100% null column** that logged "Added
  dti_ratio" and got quietly imputed downstream.
- **There is no credit limit either.** Utilization resolved to
  `existing_credits` (a count of 1-4) divided by `credit_amount` (this loan's
  size), giving a column ranging 5e-05 to 0.01 and named after a ratio it did
  not compute.

Removing both leaves logistic regression exactly where it was (0.7827 to
0.7828) and *raises* gradient boosting by more than a point (0.7730 to 0.7844).
That is what a feature contributing nothing looks like once you measure it: at
best inert, at worst something a flexible model chases the noise in. They are
skipped with a warning now rather than fabricated.

**A feature that cannot be computed must be absent, so its absence is visible. A
trustworthy name on an empty column is worse than no column.**

What survives is `loan_burden` (amount x duration) and `age_group`. Two
features, honestly earned - and `loan_burden` lands third by SHAP importance.

## 6. Explaining a rejection

An adverse action notice has to name reasons, so SHAP is not a nice-to-have
here. It is the part that makes the model legal to deploy.

It was also broken, silently, for a long time. `shap.Explainer(classifier, X)`
passes `X` as a *background* dataset, which selects the interventional
perturbation path, and on this gradient-boosted model that path failed its own
additivity check on every run:

```
sum of the SHAP values was -2.239625, while the model output was -2.379934
```

The exception was caught, logged at WARNING, and the pipeline went on to print
`[OK] Full pipeline completed successfully!` and exit 0. Meanwhile a
`shap_importance.png` from an older run sat in `artifacts/figures/` looking
perfectly current. Nothing anywhere said the regulatory feature had not run.

Two changes. `TreeExplainer` with no background uses the exact path-dependent
algorithm, which is additive by construction and faster. And a SHAP failure is
now carried into `evaluation_report.md` as a **NOT COMPUTED** block naming the
exception and warning that the figure on disk is stale, plus a `[WARN]` line on
the console.

**A silent failure in the explainability layer is worse than a crash. A crash
you fix; this one ships.**

## 7. Run it

### Clone

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/machine-learning/credit-risk-pipeline
```

### Set up with uv

[uv](https://docs.astral.sh/uv/) resolves and installs in seconds and keeps the
environment inside the project.

```bash
pip install uv

uv venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell or cmd

uv pip install -r requirements.txt
```

Plain `pip install -r requirements.txt` works identically. Python 3.10+.

### The pipeline, in order

```bash
python run.py                # everything, start to finish            (~25s)

python run.py clean          # load and clean the German Credit data  (~2s)
python run.py eda            # charts into artifacts/figures/         (~5s)
python run.py features       # loan_burden, age_group, drop personal_status
python run.py train          # LR + GradientBoosting, 5-fold CV       (~15s)
python run.py evaluate       # threshold, SHAP, written report        (~8s)
python run.py serve          # FastAPI endpoint (needs a trained model)
```

`run.py` needs **no install of this project** and runs from any working
directory. Stages run as a prefix: asking for `train` runs `clean`, `eda` and
`features` first, because each consumes what the last one produced.

`make run` and friends still work if you have `make`. It is not the documented
path, because `make` is not installed on a default Windows box - and the
Makefile's own setup target used to hardcode `.venv/Scripts/pip`, a Windows path
inside a Unix-only tool, so it worked on neither platform.

### Run the tests

```bash
pytest -p no:warnings          # 56 tests, no install needed
```

## 8. The notebook

`notebooks/credit-risk-pipeline.ipynb` walks the whole pipeline end to end in 38
cells and 5 charts, and is **the recommended starting point if you are new to
this.**

**On Google Colab**, no install at all:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/projects/machine-learning/credit-risk-pipeline/notebooks/credit-risk-pipeline.ipynb)

**On Kaggle**, published as a notebook you can copy and edit:

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/genieincodebottle/credit-risk-pipeline)

**Locally**, after the setup above:

```bash
jupyter lab notebooks/credit-risk-pipeline.ipynb
```

Run the cells one at a time rather than Run All. The notebook builds in order -
data quality, EDA, feature engineering, dropping the protected column,
preprocessing, training, cost-sensitive evaluation, SHAP, PSI drift - and each
section assumes you read the markdown above it.

**Its numbers will not match `src/`, on purpose.** The notebook trains a
class-weighted logistic regression on a slightly different feature set, so its
cost-minimising threshold is an interior **0.15** rather than the corner at 0.01
that `src/` finds, and honest tuning is worth $44 there against $527 here. Same
objective, different model, different answer - which is exactly why section 3
tells you to print what a threshold *does* rather than trusting its cost alone.

## 9. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `make: command not found` | `make` is not installed on Windows | use `python run.py`, which is the documented path |
| `ModuleNotFoundError` on first run | dependencies not installed | `pip install -r requirements.txt` - `run.py` names the missing package for you |
| `SHAP analysis FAILED` in the log | a shap / numba version conflict | the report says NOT COMPUTED rather than going quiet; `pip install -U shap numba llvmlite` |
| `[FAIL] No trained model` from `serve` | serving before training | `python run.py` once, then `python run.py serve` |
| the threshold looks absurdly low | it is - see [section 3](#3-the-threshold-is-a-corner-and-the-pipeline-says-so) | expected, and warned about on every run |
| `FutureWarning: observed=False` from `eda.py` | pandas 2.x groupby default | cosmetic, does not affect results |
| numbers differ slightly from this README | sklearn version differences in the CV splits | AUC should land within about 0.005 of 0.78 |

## 10. Layout

```
credit-risk-pipeline/
├── run.py                  # zero-install entry point - start here
├── main.py                 # pipeline orchestration, stage by stage
├── configs/base.yaml       # every parameter, including the 10:1 cost ratio
├── src/
│   ├── data_loader.py      # load, clean, config
│   ├── eda.py              # exploratory charts
│   ├── features.py         # loan_burden, age_group, drop_protected_attributes
│   ├── models.py           # LR + GBC, 5-fold CV, out-of-fold probabilities
│   ├── evaluate.py         # cost curve, threshold, SHAP, report
│   ├── monitor.py          # PSI drift detection
│   └── serve.py            # FastAPI endpoint
├── tests/                  # 56 tests
├── notebooks/              # the notebook, plus Kaggle metadata
├── artifacts/
│   ├── figures/            # 7 charts
│   └── results/            # best_model.joblib, threshold.json, report
└── model_card.md           # training data, performance, limitations
```

## 11. Serve the model

```bash
python run.py               # train first; writes artifacts/results/threshold.json
python run.py serve         # http://localhost:8000
```

The API loads the tuned threshold from `threshold.json` at startup rather than
hardcoding 0.5, so the decision boundary the evaluation chose is the one that
actually serves.

Given section 3, that threshold rejects most applicants. This is the honest
consequence of the configured cost ratio, and the reason a real deployment would
constrain approval rate in `configs/base.yaml` before going anywhere near
production.

## 12. Honest limitations

- **1,000 rows.** Every number here carries real sampling error. The gap between
  the two models is 0.001 AUC, which is noise, not a finding.
- **The cost ratio is asserted, not measured.** 10:1 is a plausible teaching
  figure. A real lender derives it from loss-given-default and margin, and the
  degenerate threshold in section 3 is downstream of it being unconstrained.
- **No fairness analysis beyond dropping one column.** Proxies for sex survive in
  `purpose`, `housing` and `job`. A deployment needs a disparate-impact audit.
- **No temporal validation.** German Credit has no dates, so there is no
  out-of-time test and no way to observe the drift that PSI monitoring exists to
  catch. `monitor.py` is demonstrated against synthetic shifts.
- **The notebook duplicates logic that also lives in `src/`** so it can stand
  alone. `src/` is what runs; change it first.

## License

MIT
