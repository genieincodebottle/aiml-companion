# Key Findings - IPL Dataset Analysis (2008-2024)

## Dataset Overview

- **Seasons covered:** 17 (2008-2024)
- **Total matches analysed:** 1,095
- **Total ball-by-ball records:** 260,920
- **Unique teams:** 17 franchises (after name standardisation)

---

## Statistical Testing

### Toss Advantage Hypothesis Test

| Metric | Value |
|--------|-------|
| Test used | Binomial test (two-sided, H0: p=0.5) |
| p-value | **0.61** |
| Significance level | 0.05 |
| Conclusion | **Fail to reject null hypothesis** |

**Interpretation:** There is no statistically significant evidence that winning the toss provides an advantage in winning the match. The observed toss-winner win rate does not deviate meaningfully from 50%.

---

## Team Performance

- Clear era-wise dominant teams identified across all 17 seasons.
- Franchise win percentages vary considerably, with perennial contenders (e.g., Chennai Super Kings, Mumbai Indians) maintaining above-average records.

## Venue Insights

- Certain venues strongly favour batting-first strategies while others favour chasing.
- Top 10 venues by match count were analysed for first-innings vs. second-innings advantage.

## Batsman Analysis

- Top performers identified by total runs, strike rate, batting average, and boundary-hitting ability (fours and sixes).
- Per-season breakdowns reveal consistency patterns among elite batsmen.

## Feature Engineering

- **15+ engineered features** including:
  - Team matchup win rates
  - Elo-like strength ratings (initial rating 1500, K-factor 32)
  - Home-ground advantage indicators
  - Rolling momentum features
  - Rivalry indices and interaction features

## Predictive Modelling

### Classification (Match Winner Prediction)

| Model | Details |
|-------|---------|
| Algorithm | Random Forest Classifier |
| Features | One-hot encoded teams, venue, toss decision + scaled numericals |
| Evaluation | Cross-validated accuracy with stratified splits |
| Class weighting | Balanced (to handle any class imbalance) |

### Regression (Run Margin Prediction)

| Model | Details |
|-------|---------|
| Algorithm | Gradient Boosting Regressor |
| Target | Win-by-runs margin |
| Loss function | Squared error |
| Evaluation | MAE, RMSE, R-squared |

---

## Data Quality Notes

- Team names standardised: *Royal Challengers Bengaluru* (2024 rename) mapped to *Royal Challengers Bangalore*; *Rising Pune Supergiants* consolidated to *Rising Pune Supergiant*.
- Missing values imputed systematically: winners for no-result matches, umpire names, cities, and result margins.
- Derived columns (`win_by_runs`, `win_by_wickets`, `toss_winner_won_match`) created during cleaning.
