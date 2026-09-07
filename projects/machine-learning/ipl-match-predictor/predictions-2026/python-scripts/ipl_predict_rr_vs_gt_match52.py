"""
Rajasthan Royals vs Gujarat Titans - Match 52, IPL 2026 - May 9
ML Prediction Script using Ensemble Models
Venue: Sawai Mansingh Stadium, Jaipur
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
    # Updated Elo as of May 9 (after 51 matches)
    # RR: 12 pts, 4th place, 6W-4L, but lost 3 of last 5, slipping. Sooryavanshi 404 runs.
    # GT: 12 pts, 5th place, 6W-4L, on 3-match winning streak (WWW). Best bowling unit.
    elo_ratings = {
        'Royal Challengers Bangalore': 1755,
        'Gujarat Titans': 1745,            # bump up: 3-match streak + best bowling
        'Mumbai Indians': 1495,
        'Punjab Kings': 1770,
        'Rajasthan Royals': 1685,          # slip: lost 3 of last 5, batting fragile after Sanju exit
        'Chennai Super Kings': 1660,
        'Sunrisers Hyderabad': 1740,
        'Delhi Capitals': 1640,
        'Kolkata Knight Riders': 1590,
        'Lucknow Super Giants': 1450,
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
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.55
    else:
        features['venue_home_win_pct'] = 0.55

    features['home_advantage'] = 1.0
    features['toss_factor'] = 1.05
    # Sawai Mansingh: par ~170-180, balanced, dew helps chase in evening (7:30 IST)
    # Recent trend: chase wins ~55-60% in evening matches
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.75,           # WWW streak
        'Mumbai Indians': 0.40,
        'Punjab Kings': 0.55,
        'Rajasthan Royals': 0.35,         # lost 3 of last 5
        'Chennai Super Kings': 0.55,
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.25,
        'Kolkata Knight Riders': 0.65,
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Rajasthan Royals'
away_team = 'Gujarat Titans'
venue = 'Sawai Mansingh Stadium'

print(f"\n{'='*60}")
print(f"MATCH 52: {home_team} vs {away_team}")
print(f"Venue: Sawai Mansingh Stadium, Jaipur")
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
    print(f"  {name.upper()} P(RR wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(RR wins): {ensemble:.1%}")
rr_prob = ensemble

print("\nContextual Adjustments (positive = favor RR):")
adjustments = 0
print("  + RR home at Sawai Mansingh (Jaipur), familiar conditions, crowd: +5%")
adjustments += 0.05
print("  + Vaibhav Sooryavanshi top scorer 404 runs in 10 games (RR's leading run-getter): +3%")
adjustments += 0.03
print("  + Riyan Parag returned to form with 90, captain confidence boost: +2%")
adjustments += 0.02
print("  + Jadeja and Sam Curran post-trade depth on Indian conditions: +2%")
adjustments += 0.02
print("  - GT on 3-match winning streak (WWW), in upswing: -6%")
adjustments -= 0.06
print("  - GT bowling unit best in IPL 2026: Siraj + Rabada + Holder + Rashid Khan: -7%")
adjustments -= 0.07
print("  - GT chasing record best in IPL since 2022, dew helps GT bat 2nd: -3%")
adjustments -= 0.03
print("  - H2H all-time: GT leads 6-3 (67% win rate vs RR): -4%")
adjustments -= 0.04
print("  - RR lost 3 of last 5, fragile batting after Sanju Samson trade to CSK: -5%")
adjustments -= 0.05
print("  - Shubman Gill leading GT runs (373) + Buttler at 3, top order strong: -2%")
adjustments -= 0.02

final_rr = max(0.10, min(0.90, rr_prob + adjustments))
final_gt = 1.0 - final_rr
print(f"\nFinal P(RR wins): {final_rr:.1%}")
print(f"Final P(GT wins): {final_gt:.1%}")

np.random.seed(42)
rr_wins = final_rr >= 0.5
winner_abbrev = 'RR' if rr_wins else 'GT'
winner_full = 'Rajasthan Royals' if rr_wins else 'Gujarat Titans'
winner_conf = int(final_rr * 100) if rr_wins else int(final_gt * 100)

# Sawai Mansingh: par ~170-180, slightly bat-friendly
if rr_wins:
    rr_score_mu, gt_score_mu = 180, 162
else:
    rr_score_mu, gt_score_mu = 162, 180
rr_score = np.random.normal(loc=rr_score_mu, scale=20, size=10000)
gt_score = np.random.normal(loc=gt_score_mu, scale=20, size=10000)
if rr_wins:
    margin = rr_score - gt_score
else:
    margin = gt_score - rr_score
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
