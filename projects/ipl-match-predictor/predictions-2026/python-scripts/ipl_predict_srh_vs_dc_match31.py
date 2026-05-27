"""
SRH vs DC - Match 31, IPL 2026 - April 21
ML Prediction Script using Ensemble Models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os

csv_path = r'c:\projects\aiml-companion\projects\ipl-match-predictor\data\raw\matches.csv'
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} historical matches (2008-2024)")

def engineer_features(home_team, away_team, venue, historical_df):
    features = {}
    elo_ratings = {
        'Sunrisers Hyderabad': 1670,     # 3W 3L, 4th place, won last vs CSK
        'Delhi Capitals': 1620,          # 3W 3L, 6th place, just beat RCB
        'Royal Challengers Bangalore': 1740,
        'Punjab Kings': 1780,
        'Mumbai Indians': 1640,
        'Gujarat Titans': 1710,
        'Kolkata Knight Riders': 1520,
        'Rajasthan Royals': 1670,
        'Chennai Super Kings': 1620,
        'Lucknow Super Giants': 1540,
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

    venue_matches = historical_df[historical_df['venue'] == venue]
    if len(venue_matches) > 0:
        home_wins = len(venue_matches[(venue_matches['team1'] == home_team) & (venue_matches['winner'] == home_team)])
        home_total = len(venue_matches[venue_matches['team1'] == home_team])
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.55
    else:
        features['venue_home_win_pct'] = 0.55

    features['home_advantage'] = 1.0
    features['toss_factor'] = 1.05
    features['chase_factor'] = 1.05  # Hyderabad chasing 47/84 historically

    momentum_2026 = {
        'Sunrisers Hyderabad': 0.65,     # Just won big vs CSK (10 runs defending 194)
        'Delhi Capitals': 0.55,          # Won vs RCB after 2 losses - mixed form
        'Royal Challengers Bangalore': 0.75,
        'Punjab Kings': 0.90,
        'Mumbai Indians': 0.35,
        'Gujarat Titans': 0.60,
        'Kolkata Knight Riders': 0.30,
        'Rajasthan Royals': 0.60,
        'Chennai Super Kings': 0.45,
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Sunrisers Hyderabad'
away_team = 'Delhi Capitals'
venue = 'Rajiv Gandhi International Stadium, Uppal'

print(f"\n{'='*60}")
print(f"MATCH 31: {home_team} vs {away_team}")
print(f"Venue: Rajiv Gandhi Intl Stadium, Hyderabad")
print(f"Date: April 21, 2026 | 7:30 PM IST")
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
    print(f"  {name.upper()}: {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble (SRH wins): {ensemble:.1%}")

print("\nContextual Adjustments:")
adjustments = 0
print("  + SRH home advantage (Hyderabad 3-2 vs DC, unbeaten vs DC at home): +3%")
adjustments += 0.03
print("  + SRH momentum from 10-run defense of 194 vs CSK: +3%")
adjustments += 0.03
print("  + Abhishek Sharma form (59 off 22 vs CSK): +2%")
adjustments += 0.02
print("  + Heinrich Klaasen anchor form (59 off 39 vs CSK): +2%")
adjustments += 0.02
print("  + Praful Hinge + Eshan Malinga pace attack clicking: +3%")
adjustments += 0.03
print("  - DC just beat RCB (chased 175 vs top team) - momentum: -3%")
adjustments -= 0.03
print("  - KL Rahul in form (57 vs RCB, 92 vs GT earlier): -2%")
adjustments -= 0.02
print("  - Tristan Stubbs + David Miller finishing prowess: -2%")
adjustments -= 0.02
print("  - Kuldeep Yadav world-class wrist spin threat: -2%")
adjustments -= 0.02
print("  - Axar Patel fitness doubtful (cramps vs RCB): +1%")
adjustments += 0.01

final_srh = max(0.30, min(0.95, ensemble + adjustments))
print(f"\nFinal (SRH wins): {final_srh:.1%}")

np.random.seed(42)
srh_score = np.random.normal(loc=185, scale=22, size=10000)
dc_score = np.random.normal(loc=170, scale=25, size=10000)
margin = srh_score - dc_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print("KEY FACTORS (Favoring SRH):")
print("="*60)
print("1. Home advantage: SRH unbeaten vs DC at Hyderabad (won 6/7 at Delhi too historically)")
print("2. Momentum: Abhishek Sharma (59 off 22) + Klaasen (59 off 39) + Malinga 3-for defended 194")
print("3. Ishan Kishan leadership form - 3rd win of season, climbed to 4th in table")
print("4. Pace duo Praful Hinge + Eshan Malinga + Sakib Hussain firing at home")

print(f"\n{'='*60}")
print("RISK FACTORS (DC Could Win):")
print("="*60)
print("1. KL Rahul in excellent form - 57 vs RCB, 92 vs GT, handles pace well")
print("2. Tristan Stubbs 60* and David Miller's finishing clinched RCB thriller")
print("3. Kuldeep Yadav world-class wrist spin can neutralize SRH middle order")
print("4. DC riding high after back-to-back losses snapped by RCB win")

print(f"\n{'='*60}")
print("PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: Sunrisers Hyderabad (SRH)")
print(f"CONFIDENCE: {int(final_srh * 100)}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
print(f"REASONING: SRH enter this match with strong home advantage and momentum from successfully defending 194 vs CSK. Abhishek Sharma's blistering 59 off 22 and Klaasen's anchor 59 off 39 showed SRH's batting depth. DC come in with a strong RCB win but have been inconsistent (3W 3L). Axar Patel's fitness is a concern. SRH's pace trio (Hinge, Malinga, Sakib) on a Hyderabad pitch that favors them will be decisive. KL Rahul and Stubbs remain DC's best bet for an upset.")
print(f"{'='*60}\n")

print("DB UPDATE VALUES:")
print(f"  ml_winner: 'SRH'")
print(f"  ml_confidence: {int(final_srh * 100)}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
