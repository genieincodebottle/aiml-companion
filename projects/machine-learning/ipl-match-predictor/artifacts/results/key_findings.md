# Key Findings - IPL Dataset Analysis (2008-2025)

Numbers below are measured on the shipped `data/raw/matches.csv` (2008-2025).
The earlier 2008-2024 file (1,095 matches) is noted where a number differs.

## Dataset Overview

- **Seasons covered:** 18 (2008-2025)
- **Total matches analysed:** 1,169 (1,161 with a decided winner, 8 no-result)
- **Total ball-by-ball records:** 260,920 (cover the 1,095 matches of 2008-2024 only)
- **Unique teams:** 15 franchises after name standardisation (the Elo model rates 15 teams; the notebook, which does not merge all renames, rates 17)

---

## Statistical Testing

### Toss Advantage Hypothesis Test

| Metric | Value |
|--------|-------|
| Test used | Binomial test (two-sided, H0: p=0.5) |
| Toss winner also won | 598 of 1,161 decided matches (51.5%) |
| p-value | **0.318** (2008-2024 file: 554 of 1,090, 50.8%, p = 0.607) |
| Significance level | 0.05 |
| Conclusion | **Fail to reject null hypothesis** |

**Interpretation:** There is no statistically significant evidence that winning the toss provides an advantage in winning the match. The observed toss-winner win rate does not deviate meaningfully from 50%.

---

## Team Performance

- Clear era-wise dominant teams identified across all 18 seasons.
- Franchise win percentages vary considerably, with perennial contenders (e.g., Chennai Super Kings, Mumbai Indians) maintaining above-average records.

## Venue Insights

- Certain venues strongly favour batting-first strategies while others favour chasing.
- Top 10 venues by match count were analysed for first-innings vs. second-innings advantage.

## Batsman Analysis

- Top performers identified by total runs, strike rate, batting average, and boundary-hitting ability (fours and sixes).
- Per-season breakdowns reveal consistency patterns among elite batsmen.

## Feature Engineering

- **20+ engineered features** including:
  - Team matchup win rates
  - Elo strength ratings (initial rating 1500, K-factor 32), ranks 1-4 in Random Forest importance
  - Home-ground advantage indicators
  - Rolling momentum features
  - Rivalry indices and interaction features

## Predictive Modelling

### Classification (Match Winner Prediction)

| Model | Details |
|-------|---------|
| Algorithm | Random Forest Classifier |
| Features | One-hot encoded teams, venue, toss decision + scaled numericals |
| Evaluation | Out-of-time split (train 2008-2023, test 2024-25, 142 matches) plus 5-fold forward-in-time CV on the training seasons only (expanding window) |
| Class weighting | Balanced (to handle any class imbalance) |

| Result (test = 2024-25) | Accuracy |
|-------------------------|----------|
| Majority-class baseline (always team 1) | 0.4789 |
| Random Forest | 0.4507 |
| XGBoost | 0.4859 |
| Neural Net | 0.5000 |
| Calibrated, recency-weighted 4-model ensemble | 0.5000 (+0.021 over baseline, within noise) |

Forward CV accuracy on 2008-2023: Random Forest 0.5290, XGBoost 0.5172, Gradient Boosting 0.5089, Logistic Regression 0.5373, Neural Net 0.5089. `main.py` picks among RF / XGBoost (Neural Net is opt-in), so the CV pick is the Random Forest, test 0.4507. With 142 test matches one standard error is about 0.042, so no model is distinguishable from the baseline.

### Regression (Run Margin Prediction)

| Model | Details |
|-------|---------|
| Algorithm | Gradient Boosting Regressor |
| Target | Win-by-runs margin |
| Loss function | Squared error |
| Evaluation | MAE, RMSE, R-squared |
| Result (5 forward folds, 463 training-season matches won by runs) | MAE 23.39, R-squared -0.467; predicting the mean margin gives MAE 20.35. On 2024-25 (68 matches): MAE 21.99, R-squared -0.186, baseline 20.81 |

---

## Data Quality Notes

- Team names standardised: *Royal Challengers Bengaluru* (2024 rename) mapped to *Royal Challengers Bangalore*; *Rising Pune Supergiants* consolidated to *Rising Pune Supergiant*.
- Missing values imputed systematically: winners for no-result matches, umpire names, cities, and result margins.
- Derived columns (`win_by_runs`, `win_by_wickets`, `toss_winner_won_match`) created during cleaning.
