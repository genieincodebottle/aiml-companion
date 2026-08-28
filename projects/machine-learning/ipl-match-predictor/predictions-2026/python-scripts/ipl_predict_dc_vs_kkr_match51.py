"""
Delhi Capitals vs Kolkata Knight Riders - Match 51, IPL 2026 - May 8
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
    # Updated Elo as of May 8 (after 50 matches)
    # DC: 8 pts, 7th place, 4W-6L, last 5: LWLLL (4 losses in last 5), KL Rahul carrying
    # KKR: 8 pts, 8th place, 3W-2L+ recent, on 3-match winning streak, games in hand
    elo_ratings = {
        'Royal Challengers Bangalore': 1755,
        'Gujarat Titans': 1730,
        'Mumbai Indians': 1495,
        'Punjab Kings': 1770,
        'Rajasthan Royals': 1700,
        'Chennai Super Kings': 1660,
        'Sunrisers Hyderabad': 1740,
        'Delhi Capitals': 1660,           # slipped: 4 losses in last 5
        'Kolkata Knight Riders': 1570,    # bump up: 3-match winning streak
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
    # Arun Jaitley: batting-friendly black soil, par 178, low dew in May
    # Recent trend at venue: 4 of 5 matches won by team batting second
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.70,
        'Gujarat Titans': 0.70,
        'Mumbai Indians': 0.25,
        'Punjab Kings': 0.50,
        'Rajasthan Royals': 0.40,
        'Chennai Super Kings': 0.40,
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.30,           # LWLLL last 5, in slide
        'Kolkata Knight Riders': 0.65,    # 3-match winning streak
        'Lucknow Super Giants': 0.10,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Delhi Capitals'
away_team = 'Kolkata Knight Riders'
venue = 'Arun Jaitley Stadium'

print(f"\n{'='*60}")
print(f"MATCH 51: {home_team} vs {away_team}")
print(f"Venue: Arun Jaitley Stadium, Delhi")
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
print("  + Home advantage at Arun Jaitley, familiar conditions: +5%")
adjustments += 0.05
print("  + KL Rahul peak form: 445 runs @ 49.44 avg, SR 181 (Orange Cap race): +4%")
adjustments += 0.04
print("  + Axar Patel leading bowler (9 wkts), spin-friendly Delhi tracks: +3%")
adjustments += 0.03
print("  + Kuldeep Yadav home advantage on slow Delhi pitch: +3%")
adjustments += 0.03
print("  + Mitchell Starc + T Natarajan new-ball threat vs KKR top order: +2%")
adjustments += 0.02
print("  - DC in slide: lost 4 of last 5, momentum vs them: -5%")
adjustments -= 0.05
print("  - KKR on 3-match winning streak (beat SRH, LSG): -4%")
adjustments -= 0.04
print("  - Axar Patel batting form crisis: 1 double-digit score in 7 innings: -2%")
adjustments -= 0.02
print("  - Rinku Singh red-hot: 83* vs LSG, 22* vs SRH (finishing matches): -3%")
adjustments -= 0.03
print("  - Narine + Varun spin twins lethal vs DC's vulnerable middle order: -3%")
adjustments -= 0.03
print("  - H2H all-time: KKR leads 19-15 (54% win rate vs DC): -2%")
adjustments -= 0.02

final_dc = max(0.10, min(0.90, dc_prob + adjustments))
final_kkr = 1.0 - final_dc
print(f"\nFinal P(DC wins): {final_dc:.1%}")
print(f"Final P(KKR wins): {final_kkr:.1%}")

np.random.seed(42)
dc_wins = final_dc >= 0.5
winner_abbrev = 'DC' if dc_wins else 'KKR'
winner_full = 'Delhi Capitals' if dc_wins else 'Kolkata Knight Riders'
winner_conf = int(final_dc * 100) if dc_wins else int(final_kkr * 100)

# Arun Jaitley scoring: par 178, batting-friendly
if dc_wins:
    dc_score_mu, kkr_score_mu = 185, 165
else:
    dc_score_mu, kkr_score_mu = 165, 185
dc_score = np.random.normal(loc=dc_score_mu, scale=20, size=10000)
kkr_score = np.random.normal(loc=kkr_score_mu, scale=20, size=10000)
if dc_wins:
    margin = dc_score - kkr_score
else:
    margin = kkr_score - dc_score
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
