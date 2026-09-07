"""
Delhi Capitals vs Chennai Super Kings - Match 48, IPL 2026 - May 5
ML Prediction Script using Ensemble Models
Venue: Arun Jaitley Stadium, Delhi
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
    # Updated Elo based on IPL 2026 standings as of May 5 (after 47 matches)
    # DC: 4W-5L, just snapped 3-match losing streak with record 226 chase vs RR (May 1)
    # CSK: 4W-5L, beat MI by 8 wkts at Chepauk (May 2)
    elo_ratings = {
        'Royal Challengers Bangalore': 1745,
        'Gujarat Titans': 1735,
        'Mumbai Indians': 1490,
        'Punjab Kings': 1780,
        'Rajasthan Royals': 1700,           # lost to DC May 1
        'Chennai Super Kings': 1645,        # won M44 vs MI
        'Sunrisers Hyderabad': 1745,
        'Delhi Capitals': 1690,             # won M43, momentum surge
        'Kolkata Knight Riders': 1515,
        'Lucknow Super Giants': 1465,
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
    # Arun Jaitley: slow surface, dew in evening, chasing slightly favored
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.20,
        'Punjab Kings': 0.55,               # lost 2 straight
        'Rajasthan Royals': 0.40,
        'Chennai Super Kings': 0.45,        # 1 win after slump
        'Sunrisers Hyderabad': 0.85,
        'Delhi Capitals': 0.55,             # snapped losing streak with big chase
        'Kolkata Knight Riders': 0.45,
        'Lucknow Super Giants': 0.15,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Delhi Capitals'
away_team = 'Chennai Super Kings'
venue = 'Arun Jaitley'

print(f"\n{'='*60}")
print(f"MATCH 48: {home_team} vs {away_team}")
print(f"Venue: Arun Jaitley Stadium, Delhi, Date: May 5, 2026")
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
    print(f"  {name.upper()} P(DC wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(DC wins): {ensemble:.1%}")
dc_prob = ensemble

print("\nContextual Adjustments (positive = favor DC):")
adjustments = 0
print("  + Home advantage at Arun Jaitley, DC 2026 home record solid: +4%")
adjustments += 0.04
print("  + DC momentum from record 226 chase vs RR (highest ever) snapped 3-loss streak: +4%")
adjustments += 0.04
print("  + DC bowling depth: Mitchell Starc + Kuldeep Yadav + T Natarajan + Axar Patel: +4%")
adjustments += 0.04
print("  + KL Rahul (wk) in form, Tristan Stubbs explosive middle order: +3%")
adjustments += 0.03
print("  + Kuldeep Yadav vs CSK middle order (Brevis, Dube) - quality wrist spin advantage: +3%")
adjustments += 0.03
print("  + CSK without MS Dhoni (per predicted XI) - significant finishing/captaincy void: +3%")
adjustments += 0.03
print("  - CSK historical H2H dominance: 20-12 all-time (62.5% win rate): -5%")
adjustments -= 0.05
print("  - Ruturaj Gaikwad in form (67* vs MI), CSK top order can post big totals: -3%")
adjustments -= 0.03
print("  - Noor Ahmad spin threat on slow Delhi surface: -3%")
adjustments -= 0.03
print("  - Both teams must-win for playoffs - cancels DC desperation edge: -2%")
adjustments -= 0.02
print("  - Sanju Samson (per predicted XI as wk) adds CSK batting firepower: -2%")
adjustments -= 0.02

final_dc = max(0.10, min(0.90, dc_prob + adjustments))
final_csk = 1.0 - final_dc
print(f"\nFinal P(DC wins):   {final_dc:.1%}")
print(f"Final P(CSK wins):  {final_csk:.1%}")

np.random.seed(42)
dc_wins = final_dc >= 0.5
winner_abbrev = 'DC' if dc_wins else 'CSK'
winner_full = 'Delhi Capitals' if dc_wins else 'Chennai Super Kings'
winner_conf = int(final_dc * 100) if dc_wins else int(final_csk * 100)

if dc_wins:
    dc_score_mu, csk_score_mu = 178, 162
else:
    dc_score_mu, csk_score_mu = 158, 175
dc_score = np.random.normal(loc=dc_score_mu, scale=20, size=10000)
csk_score = np.random.normal(loc=csk_score_mu, scale=20, size=10000)
if dc_wins:
    margin = dc_score - csk_score
else:
    margin = csk_score - dc_score
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
