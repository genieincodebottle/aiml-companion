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
| ROC-AUC | ~0.76 | ~0.79 |
| Recall | ~0.65 | ~0.62 |
| F1 | ~0.55 | ~0.57 |

*Metrics from 5-fold stratified cross-validation. Actual values depend on random seed and preprocessing.*

## Cost-Sensitive Threshold

- Default threshold: 0.35 (tuned for 10:1 FN/FP cost ratio)
- Rationale: Missing a defaulter (FN) costs ~10x more than denying a good borrower (FP)
- Optimal threshold determined by minimizing: Cost = FN * 10 + FP * 1

## Limitations

- **Small dataset**: Only 1,000 samples limits generalization
- **Historical bias**: Dataset from 1994 German banking, may reflect outdated patterns
- **No protected attributes audit**: Fairness analysis not included in base pipeline
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