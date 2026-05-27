"""
Gujarat Titans vs Sunrisers Hyderabad - Match 56, IPL 2026 - May 12
ML Prediction Script using Ensemble Models
Venue: Narendra Modi Stadium, Ahmedabad
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

csv_path = r'c:\projects\aiml-companion\projects\ipl-match-predictor\data\raw\matches.csv'
df = pd.read_csv(csv_path)
df['team1'] = df['team1'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['team2'] = df['team2'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['winner'] = df['winner'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
print(f"Loaded {len(df)} historical matches (2008-2024)")

def engineer_features(home_team, away_team, venue, historical_df):
    features = {}
    # Updated Elo as of May 12 (after 55 matches)
    # GT: 14 pts (3rd, level on points with top), 4-match winning streak (W vs DC, MI ouster, vs RR W in match 52)
    # SRH: 14 pts (2nd on NRR), won 6 of last 7, lost to KKR most recently. Top in NRR.
    # RCB also 14 pts (top by NRR currently)
    elo_ratings = {
        'Royal Challengers Bangalore': 1775,
        'Gujarat Titans': 1775,            # bump: 4-match streak + dominant H2H vs SRH at Ahmedabad
        'Mumbai Indians': 1500,
        'Punjab Kings': 1760,
        'Rajasthan Royals': 1675,
        'Chennai Super Kings': 1650,
        'Sunrisers Hyderabad': 1770,       # strong: top of table by NRR, 6 of last 7 W
        'Delhi Capitals': 1635,
        'Kolkata Knight Riders': 1610,     # bump: beat SRH recently
        'Lucknow Super Giants': 1440,
    }
    home_elo = elo_ratings.get(home_team, 1600)
    away_elo = elo_ratings.get(away_team, 1600)
    features['elo_diff'] = home_elo - away_elo
    features['home_elo'] = home_elo
    features['away_elo'] = away_elo

    h2h = historical_df[
        ((historical_df['team1'] == home_team) & (historical_df['team2'] == away_team)) |
        ((historical_df['team1'] == away_team) & (historical_df['team2'] == home_team))
    ].tail(5)
    if len(h2h) > 0:
        home_h2h_wins = len(h2h[h2h['winner'] == home_team])
        features['h2h_home_win_pct'] = home_h2h_wins / len(h2h)
        features['h2h_matches'] = len(h2h)
    else:
        features['h2h_home_win_pct'] = 0.5
        features['h2h_matches'] = 0

    venue_matches = historical_df[historical_df['venue'].str.contains(venue, na=False, case=False)]
    if len(venue_matches) > 0:
        home_wins = len(venue_matches[(venue_matches['team1'] == home_team) & (venue_matches['winner'] == home_team)])
        home_total = len(venue_matches[venue_matches['team1'] == home_team])
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.58
    else:
        features['venue_home_win_pct'] = 0.58

    features['home_advantage'] = 1.0
    features['toss_factor'] = 1.05
    # Narendra Modi: par ~180-200 in 2025, dew helps chase in evening (7:30 IST)
    # Chasing team has slight edge but home GT bowling unit excellent at defending
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.80,           # 4-match streak, riding high
        'Mumbai Indians': 0.45,
        'Punjab Kings': 0.55,
        'Rajasthan Royals': 0.30,
        'Chennai Super Kings': 0.50,
        'Sunrisers Hyderabad': 0.70,      # 4W-1L last 5
        'Delhi Capitals': 0.25,
        'Kolkata Knight Riders': 0.65,
        'Lucknow Super Giants': 0.25,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Gujarat Titans'
away_team = 'Sunrisers Hyderabad'
venue = 'Narendra Modi Stadium'

print(f"\n{'='*60}")
print(f"MATCH 56: {home_team} vs {away_team}")
print(f"Venue: Narendra Modi Stadium, Ahmedabad")
print(f"{'='*60}\n")

match_features = engineer_features(home_team, away_team, venue, df)
print("Feature Engineering:")
for k, v in match_features.items():
    print(f"  {k}: {v}")

train_df = df.dropna(subset=['winner'])
X_cols, X_data = [], []
for _, row in train_df.iterrows():
    feats = engineer_features(row['team1'], row['team2'], row['venue'], df)
    if not X_data:
        X_cols = list(feats.keys())
    X_data.append([feats[k] for k in X_cols])

X = np.array(X_data)
y = (train_df['team1'] == train_df['winner']).astype(int).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
match_scaled = scaler.transform(np.array([[match_features[k] for k in X_cols]]))

print(f"\nTraining on {len(X)} matches...")
models = {
    'rf': RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
    'xgb': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42),
    'gb': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'lr': LogisticRegression(max_iter=500, class_weight='balanced')
}
predictions = {}
for name, model in models.items():
    model.fit(X_scaled, y)
    pred = model.predict_proba(match_scaled)[0][1]
    predictions[name] = pred
    print(f"  {name.upper()} P(GT wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(GT wins): {ensemble:.1%}")
gt_prob = ensemble

print("\nContextual Adjustments (positive = favor GT):")
adjustments = 0
# Model is overfitting on H2H tail (5-0 in dataset). Pull base back toward parity
# given both teams have 14 pts and SRH leads NRR.
print("  - Base regression: SRH equal on points (14), tops NRR table - genuinely top-tier: -15%")
adjustments -= 0.15
print("  + GT home at Narendra Modi Stadium, Ahmedabad, familiar deck and crowd: +3%")
adjustments += 0.03
print("  + GT unbeaten vs SRH in Ahmedabad since IPL 2023 (3-0 at home): +5%")
adjustments += 0.05
print("  + GT on 4-match winning streak, riding momentum: +3%")
adjustments += 0.03
print("  + Shubman Gill captain in elite form: 462 runs in 10, 4th-highest scorer: +2%")
adjustments += 0.02
print("  + GT bowling unit balance: Siraj + Rabada (pace) + Rashid (spin) + Sundar: +3%")
adjustments += 0.03
print("  - Heinrich Klaasen 494 runs (more than Gill), in superb touch: -3%")
adjustments -= 0.03
print("  - SRH have crossed 200 runs 8 times this season - explosive top order: -4%")
adjustments -= 0.04
print("  - SRH won 6 of last 7, only loss to KKR - equally hot momentum: -3%")
adjustments -= 0.03
print("  - Eshan Malinga leads SRH wickets (16), reverse swing threat at death: -2%")
adjustments -= 0.02
print("  - Abhishek Sharma + Travis Head opening pair attacks powerplay: -2%")
adjustments -= 0.02
print("  - Dew at Ahmedabad helps chase, toss-winner likely bowls first - neutralises home edge: -2%")
adjustments -= 0.02

final_gt = max(0.10, min(0.90, gt_prob + adjustments))
final_srh = 1.0 - final_gt
print(f"\nFinal P(GT wins): {final_gt:.1%}")
print(f"Final P(SRH wins): {final_srh:.1%}")

np.random.seed(42)
gt_wins = final_gt >= 0.5
winner_abbrev = 'GT' if gt_wins else 'SRH'
winner_full = 'Gujarat Titans' if gt_wins else 'Sunrisers Hyderabad'
winner_conf = int(final_gt * 100) if gt_wins else int(final_srh * 100)

# Narendra Modi: par ~180-200, high-scoring deck
if gt_wins:
    gt_score_mu, srh_score_mu = 188, 170
else:
    gt_score_mu, srh_score_mu = 170, 188
gt_score = np.random.normal(loc=gt_score_mu, scale=20, size=10000)
srh_score = np.random.normal(loc=srh_score_mu, scale=20, size=10000)
if gt_wins:
    margin = gt_score - srh_score
else:
    margin = srh_score - gt_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print(f"PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: {winner_full} ({winner_abbrev})")
print(f"CONFIDENCE: {winner_conf}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
print(f"{'='*60}\n")
print("DB UPDATE VALUES:")
print(f"  ml_winner: '{winner_abbrev}'")
print(f"  ml_confidence: {winner_conf}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
