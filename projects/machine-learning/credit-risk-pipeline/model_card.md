# Model Card: Credit Risk Default Prediction

## Model Details

- **Model type**: Logistic Regression (primary) + Gradient Boosting Classifier (challenger)
- **Framework**: scikit-learn 1.3+
- **Training data**: German Credit Dataset (OpenML ID 31, 1000 samples)
- **Task**: Binary classification (default vs. no default)
- **Version**: 0.1.0
- **License**: MIT

## Intended Use

- **Primary use**: Educational demonstration of end-to-end ML pipeline for credit risk
- **Intended users**: ML learners, data science students
- **Out of scope**: Production credit decisions, regulatory submissions

## Training Data

- **Source**: UCI German Credit Dataset via OpenML
- **Size**: 1,000 borrowers, 20 features
- **Class distribution**: ~70% no default, ~30% default
- **Features**: Duration, credit amount, employment, housing, savings, checking account status, purpose, and more

## Evaluation Metrics

| Metric | Logistic Regression | Gradient Boosting |
|--------|-------------------|-------------------|
| ROC-AUC | 0.783 (+/- 0.025) | 0.784 (+/- 0.025) |
| Recall | 0.700 | 0.493 |
| F1 | 0.595 | 0.557 |

*Metrics from 5-fold stratified cross-validation, measured on the current pipeline.*

**The 0.001 ROC-AUC gap is noise, not a result.** It is far inside the 0.025
cross-validation standard deviation, and which model comes out ahead changes with
the seed and the feature set - the notebook usually picks Logistic Regression,
the CLI usually picks Gradient Boosting. Logistic Regression is the one to ship
regardless, because ECOA adverse-action notices are easier to defend from
coefficients than from an ensemble.

## Cost-Sensitive Threshold

- Tuned threshold: **0.01**, chosen by minimising Cost = FN * 10 + FP * 1 over
  out-of-fold predictions, and written to `artifacts/results/threshold.json`
- Rationale: missing a defaulter (FN) is priced at 10x denying a good borrower (FP)

**This threshold is a degenerate corner and must not be deployed as-is.** At 0.01
the model flags about 91% of applicants and rejects 612 of the 700 borrowers who
would have repaid. It minimises the stated cost because the objective contains no
approval-rate or capital constraint - not because it is a sensible lending policy.
The pipeline warns about this on every run rather than hiding it. A real
deployment adds the missing constraint to `configs/base.yaml` first.

## Limitations

- **Small dataset**: Only 1,000 samples limits generalization
- **Historical bias**: Dataset from 1994 German banking, may reflect outdated patterns
- **Fairness stops at one dropped column**: `personal_status` encodes sex and is
  dropped by default (`drop_protected`), which costs about 0.006 AUC. Proxies
  survive in `purpose`, `housing` and `job`, and there is **no disparate-impact
  audit on outcomes**, which a real deployment would require
- **Feature simplicity**: Real credit scoring uses 100+ features (bureau data, payment history)
- **Not for production**: This is an educational project, not a production credit scoring system

## Ethical Considerations

- Credit scoring directly affects people's access to financial services
- Models trained on historical data may perpetuate existing biases
- ECOA (Equal Credit Opportunity Act) requires adverse action reasons for denials
- FCRA (Fair Credit Reporting Act) requires model interpretability
- Logistic Regression chosen as primary model specifically for regulatory interpretability

## Monitoring

- PSI (Population Stability Index) for distribution drift detection
- Data quality checks for incoming predictions
- Thresholds: PSI < 0.10 (stable), 0.10-0.25 (investigate), > 0.25 (retrain)