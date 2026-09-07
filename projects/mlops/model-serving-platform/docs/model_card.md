# Model Card: Iris Classifier (Demo)

## Model Details
- **Type**: RandomForestClassifier (sklearn)
- **Version**: v1.0.0
- **Purpose**: Demo model for MLOps infrastructure capstone
- **Training Data**: Iris dataset (150 samples, 4 features)

## Intended Use
- Demonstrate model serving, monitoring, and CI/CD infrastructure
- NOT a production model - the infrastructure IS the deliverable

## Performance
- Accuracy: 0.97 on test set
- Inference: <5ms per prediction

## Limitations
- Iris is a toy dataset - this model demonstrates infrastructure, not ML
- No fairness analysis needed (non-sensitive classification)
