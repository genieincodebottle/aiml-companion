"""
Mumbai Indians vs Lucknow Super Giants - Match 47, IPL 2026 - May 4
ML Prediction Script using Ensemble Models
Venue: Wankhede Stadium, Mumbai
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
    # Updated Elo based on IPL 2026 standings as of May 4 (after 46 matches)
    # MI 9th (2W-7L, on 3-match losing streak)
    # LSG 10th (2W-7L, on 5-match losing streak)
    elo_ratings = {
        'Royal Challengers Bangalore': 1745,
        'Gujarat Titans': 1730,           # won match 46
        'Mumbai Indians': 1495,           # 2W-7L, 9th, 3 straight losses
        'Punjab Kings': 1790,             # lost match 46 but still top in standings
        'Rajasthan Royals': 1715,
        'Chennai Super Kings': 1620,
        'Sunrisers Hyderabad': 1740,      # lost match 45
        'Delhi Capitals': 1665,
        'Kolkata Knight Riders': 1510,    # won match 45, broke SRH streak
        'Lucknow Super Giants': 1465,     # 2W-7L, 10th, 5 straight losses
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
    # Wankhede: high-scoring, sea breeze, heavy dew - chasing strongly favored in evening
    features['chase_factor'] = 1.10

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.20,           # 2W-7L, 3 straight losses
        'Punjab Kings': 0.80,
        'Rajasthan Royals': 0.55,
        'Chennai Super Kings': 0.40,
        'Sunrisers Hyderabad': 0.75,
        'Delhi Capitals': 0.55,
        'Kolkata Knight Riders': 0.40,
        'Lucknow Super Giants': 0.15,     # 2W-7L, 5 straight losses, last in table
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Mumbai Indians'
away_team = 'Lucknow Super Giants'
venue = 'Wankhede'

print(f"\n{'='*60}")
print(f"MATCH 47: {home_team} vs {away_team}")
print(f"Venue: Wankhede Stadium, Mumbai, Date: May 4, 2026")
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
    print(f"  {name.upper()} P(MI wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(MI wins): {ensemble:.1%}")
mi_prob = ensemble

print("\nContextual Adjustments (positive = favor MI):")
adjustments = 0
print("  + Wankhede home advantage - MI know dew + breeze, batting paradise: +4%")
adjustments += 0.04
print("  + Bumrah + Boult new ball pair best in IPL, exposes LSG fragile top order: +3%")
adjustments += 0.03
print("  + Tilak Varma in extraordinary form (101* off 45 vs GT recently): +2%")
adjustments += 0.02
print("  + Suryakumar Yadav at Wankhede - he averages exceptionally at home: +2%")
adjustments += 0.02
print("  + LSG on 5-match losing streak (longer than MI's 3) - more demoralized: +2%")
adjustments += 0.02
print("  - MI in worst form in years (2W-7L, 9th), entire batting unit misfiring: -10%")
adjustments -= 0.10
print("  - LSG H2H dominance: 6-2 (75%) vs MI all-time, mental edge: -5%")
adjustments -= 0.05
print("  - Rishabh Pant + Nicholas Pooran + Marsh + Markram - LSG batting can explode: -4%")
adjustments -= 0.04
print("  - Hardik Pandya in poor form (146 runs, 4 wkts), captaincy under scrutiny: -3%")
adjustments -= 0.03
print("  - Mayank Yadav pace + Ravi Bishnoi spin (7 wkts vs MI) genuine threat: -3%")
adjustments -= 0.03
print("  - Both teams must-win, neutralizes MI 'desperation' edge: -2%")
adjustments -= 0.02

final_mi = max(0.10, min(0.90, mi_prob + adjustments))
final_lsg = 1.0 - final_mi
print(f"\nFinal P(MI wins):   {final_mi:.1%}")
print(f"Final P(LSG wins):  {final_lsg:.1%}")

np.random.seed(42)
mi_wins = final_mi >= 0.5
winner_abbrev = 'MI' if mi_wins else 'LSG'
winner_full = 'Mumbai Indians' if mi_wins else 'Lucknow Super Giants'
winner_conf = int(final_mi * 100) if mi_wins else int(final_lsg * 100)

if mi_wins:
    mi_score_mu, lsg_score_mu = 192, 175
else:
    mi_score_mu, lsg_score_mu = 168, 188
mi_score = np.random.normal(loc=mi_score_mu, scale=20, size=10000)
lsg_score = np.random.normal(loc=lsg_score_mu, scale=22, size=10000)
if mi_wins:
    margin = mi_score - lsg_score
else:
    margin = lsg_score - mi_score
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
