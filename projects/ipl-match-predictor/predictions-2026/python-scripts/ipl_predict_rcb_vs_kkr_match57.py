"""
Royal Challengers Bangalore vs Kolkata Knight Riders - Match 57, IPL 2026 - May 13
ML Prediction Script using Ensemble Models
Venue: Shaheed Veer Narayan Singh International Stadium, Raipur (RCB secondary home)
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
    # Updated Elo as of May 13 (post match 56)
    # RCB: 14 pts (#1 by NRR +1.103), 7W-4L, latest W vs MI by 2 wickets at Raipur
    # KKR: 9 pts (#7), 4W-6L, latest W vs DC with Finn Allen 100*(47), playoff hopes faint
    elo_ratings = {
        'Royal Challengers Bangalore': 1790,   # top of table by NRR
        'Gujarat Titans': 1770,
        'Mumbai Indians': 1480,                # eliminated, 6 pts/11
        'Punjab Kings': 1735,                  # 4-match losing streak
        'Rajasthan Royals': 1660,
        'Chennai Super Kings': 1640,
        'Sunrisers Hyderabad': 1765,
        'Delhi Capitals': 1645,                # beat PBKS in chase of 211
        'Kolkata Knight Riders': 1640,         # bump: beat DC + SRH recently
        'Lucknow Super Giants': 1430,
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

    features['home_advantage'] = 0.95   # Raipur is RCB secondary venue, not Chinnaswamy
    features['toss_factor'] = 1.05
    # Raipur: par ~165-180, dew helps chase in evening (7:30 IST)
    features['chase_factor'] = 1.10

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.72,   # 4W in last 5 (LSG L, then strong run)
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.40,
        'Punjab Kings': 0.30,                  # 4-match losing streak
        'Rajasthan Royals': 0.35,
        'Chennai Super Kings': 0.55,
        'Sunrisers Hyderabad': 0.65,
        'Delhi Capitals': 0.55,
        'Kolkata Knight Riders': 0.65,         # 2 of last 3 wins, Allen century hot
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Royal Challengers Bangalore'
away_team = 'Kolkata Knight Riders'
venue = 'Shaheed Veer Narayan Singh Stadium'

print(f"\n{'='*60}")
print(f"MATCH 57: {home_team} vs {away_team}")
print(f"Venue: Shaheed Veer Narayan Singh Stadium, Raipur")
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
    print(f"  {name.upper()} P(RCB wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(RCB wins): {ensemble:.1%}")
rcb_prob = ensemble

print("\nContextual Adjustments (positive = favor RCB):")
adjustments = 0
print("  + Model H2H tail dominated by 2019-2023 squads (different RCB) - regress to mean: +10%")
adjustments += 0.10
print("  + RCB top of table (14 pts/11, NRR +1.103) vs KKR 9 pts/10 (#7): +5%")
adjustments += 0.05
print("  + RCB nominal home at Raipur, won here vs MI off the last ball: +3%")
adjustments += 0.03
print("  + Kohli currently top Orange Cap contender (81 vs GT, 49 vs LSG impact): +3%")
adjustments += 0.03
print("  + Patidar (capt), Tim David, Jitesh, Bethell - deep RCB batting lineup: +2%")
adjustments += 0.02
print("  + Hazlewood + Bhuvneshwar + Krunal give RCB balanced bowling at Raipur: +3%")
adjustments += 0.03
print("  + RCB have already qualified for playoffs - higher confidence: +2%")
adjustments += 0.02
print("  - Finn Allen smashed 100*(47) vs DC last out - destructive top order: -3%")
adjustments -= 0.03
print("  - Varun Chakravarthy + Sunil Narine spin twin threat: -2%")
adjustments -= 0.02
print("  - KKR must-win mode, playing for survival - higher intensity: -2%")
adjustments -= 0.02
print("  - Cameron Green allrounder gives KKR death overs depth: -1%")
adjustments -= 0.01

final_rcb = max(0.10, min(0.90, rcb_prob + adjustments))
final_kkr = 1.0 - final_rcb
print(f"\nFinal P(RCB wins): {final_rcb:.1%}")
print(f"Final P(KKR wins): {final_kkr:.1%}")

np.random.seed(42)
rcb_wins = final_rcb >= 0.5
winner_abbrev = 'RCB' if rcb_wins else 'KKR'
winner_full = 'Royal Challengers Bangalore' if rcb_wins else 'Kolkata Knight Riders'
winner_conf = int(final_rcb * 100) if rcb_wins else int(final_kkr * 100)

# Raipur: par ~165-180, batting friendly, dew helps chase
if rcb_wins:
    rcb_score_mu, kkr_score_mu = 182, 162
else:
    rcb_score_mu, kkr_score_mu = 162, 182
rcb_score = np.random.normal(loc=rcb_score_mu, scale=20, size=10000)
kkr_score = np.random.normal(loc=kkr_score_mu, scale=20, size=10000)
if rcb_wins:
    margin = rcb_score - kkr_score
else:
    margin = kkr_score - rcb_score
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
