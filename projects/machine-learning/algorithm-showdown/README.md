# Algorithm Showdown: Medical Diagnostic Classifier

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen)

Build a medical diagnostic classifier that balances catching cancer against false alarms, with full explainability for regulatory review -- and that reports the honest recall rather than the one it was tuned to produce.

## Architecture

```
UCI Breast Cancer Data (569 samples, 30 features)
    |
    v
+-----------------------------------+
|   Data Pipeline                   |
|   StandardScaler + StratifiedKFold(5) |
+-----------------------------------+
    |
    v
+-----------------------------------+
|   6 Competing Algorithms          |
|   LogReg | SVM | KNN | DT | RF | XGB |
+-----------------------------------+
    |
    v
+-----------------------------------+
|   Cost-Sensitive Evaluation       |
|   Threshold tuned for 95% recall  |
+-----------------------------------+
    |
    v
+-----------------------------------+
|   SHAP Explainability Report      |
|   Global importance + top-3 local |
+-----------------------------------+
```

## Problem Statement

Cancer screening demands high recall -- **missing a malignant tumor is far worse than a false alarm**. This project systematically compares 6 ML algorithms on real clinical data, tunes the decision threshold for a **95% malignant-recall** target, and generates SHAP explainability reports suitable for regulatory review.

The part worth staying for is what happens when you measure that 95% honestly. It does not hold.

## Approach

1. **Baseline**: Logistic Regression with StandardScaler -- establishes the accuracy/recall floor
2. **Competition**: 5 additional algorithms (SVM, KNN, Decision Tree, Random Forest, XGBoost) evaluated with identical StratifiedKFold(5) cross-validation
3. **Selection**: the winner is read off that comparison by AUC -- the right metric here precisely *because* no threshold has been chosen yet
4. **Threshold Tuning**: the selected model's threshold is moved off 0.5 to reach 95% malignant recall, tuned on a **validation** split and reported on a **test** split that neither the model nor the threshold has seen
5. **Explainability**: SHAP generates per-patient and global feature importance reports

## Results

5-fold StratifiedKFold on all 569 samples:

| Algorithm | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.974 | 0.968 | **0.992** | 0.979 | **0.995** |
| **SVM (RBF)** | **0.977** | **0.976** | 0.989 | **0.982** | 0.995 |
| KNN (k=5) | 0.963 | 0.957 | 0.986 | 0.971 | 0.985 |
| Decision Tree | 0.928 | 0.932 | 0.958 | 0.944 | 0.907 |
| Random Forest | 0.954 | 0.962 | 0.966 | 0.964 | 0.990 |
| XGBoost | 0.961 | 0.968 | 0.972 | 0.969 | 0.994 |

**No algorithm wins every column, and the gaps are small.** SVM takes accuracy,
precision and F1; logistic regression takes recall and AUC. Five of the six sit
within 2 accuracy points of each other, and the boosted model -- the one you
would reach for by reflex -- is fifth. On 569 clean, well-separated samples the
algorithm is close to irrelevant, and a 30-year-old linear model is on the
podium. Spend the effort on the decision rule instead, which is what the rest of
this project does.

### The number that was never a measurement

The threshold is the actual deliverable here: it is what decides whether a
patient gets called back. It is also a **parameter fitted to data**, and this
project used to fit it on the same rows it then reported.

That makes the headline circular. The threshold is *defined* as the one that
reaches 95% recall on those rows, so 95% recall is what gets printed --
regardless of whether the model is any good. Across 30 random seeds that number
had a standard deviation of **0.0005**. It was not an estimate of anything; it
was the target echoed back.

Tuned on a validation split and reported on an untouched test split:

| | recall, 30 seeds | sd | worst seed |
|---|---|---|---|
| tuned on the reported rows | 0.9528 | **0.0005** | 0.9524 |
| tuned on validation, reported on test | 0.9717 | **0.0367** | **0.8571** |

The honest version is *more variable*, and that variability is the finding. On
42 malignant test cases, 0.857 recall means **six missed cancers**. Precision
moves in the direction you would expect too: 0.917 in-sample against 0.833 held
out.

On the default seed the pipeline now prints:

```
Threshold tuned on VALIDATION for 95% malignant recall: 0.759
  validation: recall 0.952  precision 1.000  (in-sample -- not a result)
  TEST (held out): recall 0.860  precision 1.000   <- the only honest number
  note: test recall missed the 95% target.
```

**The shipped screening threshold does not meet its stated recall.** That is the
real state of this model, and the two-split design could not have told you --
it would have printed 0.95 and looked finished.

### What this does not fix

Tuning on 5-fold out-of-fold predictions instead of a 20% holdout was tried and
is *not* better here (test-recall sd 0.0316 vs 0.0344 over 30 seeds), so the
simpler design stays. The remaining spread is not a tuning-method problem: with
42 malignant cases in the test set, **one case is 2.4 recall points**. At this
dataset size a recall target cannot be pinned tighter than a few points by any
method, and a screening programme that needs a guaranteed 95% needs more data,
not a cleverer split.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/machine-learning/algorithm-showdown
pip install -r requirements.txt

# Run the full pipeline (train -> evaluate -> explain)
make all

# Or run individual stages
make train       # Train 6 algorithms and compare
make evaluate    # Evaluate saved model with threshold analysis
make explain     # Generate SHAP explainability report

# Run tests
make test

# Or use the pipeline script directly
bash scripts/run_pipeline.sh          # Full pipeline
bash scripts/run_pipeline.sh train    # Train only
```

## Project Structure

```
algorithm-showdown/
|-- configs/
|   +-- base.yaml               # Hyperparameters, CV settings, threshold target
|-- src/
|   |-- __init__.py              # Package marker
|   |-- train.py                 # 6-algorithm comparison pipeline
|   |-- evaluate.py              # Confusion matrices, PR curves, threshold analysis
|   +-- explain.py               # SHAP waterfall + global importance
|-- tests/
|   |-- __init__.py
|   +-- test_pipeline.py         # pytest: data loading, fit, predict, threshold
|-- notebooks/
|   +-- Algorithm_Showdown.ipynb # Interactive walkthrough
|-- artifacts/
|   |-- models/                  # Saved model artifacts (.gitkeep)
|   +-- results/
|       |-- comparison_table.md  # Pre-generated metrics table
|       +-- shap_summary.md      # Pre-generated SHAP report
|-- docs/
|   |-- model_card.md            # Google Model Card for the classifier
|   +-- experiment_log.csv       # Experiment progression tracking
|-- scripts/
|   +-- run_pipeline.sh          # Shell script: train -> evaluate -> explain
|-- .gitignore
|-- Makefile                     # make train / evaluate / explain / test / all
|-- requirements.txt             # Pinned dependencies
+-- README.md
```

## Experiment Log

| Experiment | What Changed | Accuracy | AUC | Notes |
|---|---|---|---|---|
| baseline_logreg | Initial LogReg baseline | 0.965 | 0.993 | Strong baseline |
| svm_rbf | Added SVM RBF | 0.968 | 0.994 | Slightly better |
| knn_k5 | KNN k=5 | 0.958 | 0.985 | Lowest performer |
| decision_tree | DTree depth=5 | 0.940 | 0.960 | Most interpretable |
| random_forest | RF n=200 | 0.963 | 0.994 | High variance |
| xgboost | XGB lr=0.1 | 0.970 | 0.995 | Best AUC |
| logreg_selected | Selected by AUC from the table | 0.974 | 0.995 | Bake-off now drives selection |
| threshold_2split | Threshold tuned on the reported set | -- | -- | **Withdrawn**: circular, sd 0.0005 |
| threshold_3split | Tuned on validation, reported on test | -- | -- | Test recall **0.860**, target 0.95 NOT met |

## Interview Guide: How to Talk About This Project

### "Walk me through this project."

"I built a cancer diagnostic classifier comparing 6 algorithms on clinical data. Two things I would lead with. First, the algorithm barely mattered -- five of six landed within two accuracy points and logistic regression won on AUC, so I stopped tuning models and started working on the decision rule. Second, and this is the part I actually learned from: the threshold that decides who gets called back is a parameter fitted to data, so I had been tuning it on the same rows I reported. That makes the headline circular -- across 30 seeds it reproduced the target with a standard deviation of 0.0005, because it was the target, not a measurement. Once I split validation off from test, held-out recall came out at 0.86 against a 0.95 target. The model is worse than I thought it was, and now I can see it."

### "What was the hardest part?"

"Realising that threshold tuning is model selection. It looks like a post-processing step -- you already have the model, you are just picking a number -- so it is easy to do it on the test set without noticing you have done anything. The tell was the variance: my recall figure was suspiciously stable across seeds, sd 0.0005, which is not what a real measurement looks like on 42 malignant cases. A number that never moves is not precise, it is circular. The trade-off itself is the easy part to justify: a false alarm costs a biopsy, a miss can be fatal, so I fix recall and accept the precision it costs."

### "What would you do differently?"

"Get more data, and for a specific reason rather than as a reflex. The test set holds 42 malignant cases, so one missed case moves recall by 2.4 points. I tried tuning the threshold on out-of-fold predictions instead of a holdout and it did not help (sd 0.0316 against 0.0344) -- the noise is coming from the size of the evaluation set, not from the tuning method, so no amount of methodology fixes it. Beyond that: a more imbalanced dataset, since 63/37 is far kinder than the 95/5 real screening data looks like; and calibration plots, because right now I can rank patients but I cannot honestly tell a clinician what a score of 0.7 means."

### "How does this scale to production?"

"The sklearn Pipeline handles preprocessing and scaling consistently. For deployment, I'd wrap the threshold-tuned XGBoost in a FastAPI endpoint, add input validation with Pydantic, and containerize with Docker. The SHAP explainability component would be critical for any regulated healthcare deployment."

### "Explain the threshold tuning to a non-technical person."

"Imagine a smoke detector. A sensitive detector catches every fire but sometimes triggers on burnt toast. A less sensitive one never false alarms but might miss a real fire. For cancer screening, we set the sensitivity to maximum -- we'd rather investigate 100 false alarms than miss 1 real case. That's what threshold tuning does."