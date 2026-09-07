"""
Punjab Kings vs Royal Challengers Bangalore - Match 61, IPL 2026 - May 17 (3:30 PM)
ML Prediction Script using Ensemble Models
Venue: Maharaja Yadavindra Singh Intl Cricket Stadium, Mullanpur (PBKS home)
Context: RCB (top of table, 16 pts, hot form, Kohli/Patidar/Bhuvi firing) vs PBKS (4th, 13 pts, 5 consecutive losses, must-win for direct playoff path)
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
    # Updated Elo as of May 16 evening (after matches 57, 58, 59, 60)
    # RCB: top, 16 pts, defending champs, Kohli 105* vs KKR + win over MI - peak form
    # PBKS: 13 pts/12, 4th, 5-match losing skid (lost to MI/DC/SRH/GT and one more)
    elo_ratings = {
        'Royal Challengers Bangalore': 1830,         # top of table, peak form, Kohli/Patidar firing
        'Gujarat Titans': 1815,                       # high but assume KKR/GT decision shifts slightly
        'Mumbai Indians': 1500,                       # beat PBKS, recovering
        'Punjab Kings': 1685,                         # 5 straight losses, sliding down
        'Rajasthan Royals': 1640,
        'Chennai Super Kings': 1700,                  # back-to-back wins over LSG, MI
        'Sunrisers Hyderabad': 1720,
        'Delhi Capitals': 1660,                       # beat PBKS recently
        'Kolkata Knight Riders': 1640,
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

    # Mullanpur: PBKS undefeated at home in 2026 (significant home advantage)
    features['home_advantage'] = 1.08
    features['toss_factor'] = 1.05
    # Mullanpur: bat-friendly, but dew not strong in afternoon 3:30 PM start (sunlight)
    # Afternoon match = bat first slight edge (no dew)
    features['chase_factor'] = 0.95

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.85,          # 4 wins in last 5, top of table
        'Gujarat Titans': 0.80,
        'Mumbai Indians': 0.55,                       # beat PBKS, mixed form
        'Punjab Kings': 0.20,                         # 5 consecutive losses, momentum gone
        'Rajasthan Royals': 0.30,
        'Chennai Super Kings': 0.70,
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.50,                       # beat PBKS recently
        'Kolkata Knight Riders': 0.50,
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Punjab Kings'
away_team = 'Royal Challengers Bangalore'
venue = 'Mullanpur'

print(f"\n{'='*60}")
print(f"MATCH 61: {home_team} vs {away_team}")
print(f"Venue: Maharaja Yadavindra Singh ICS, Mullanpur")
print(f"Time: 3:30 PM IST (afternoon)")
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
    print(f"  {name.upper()} P(PBKS wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(PBKS wins): {ensemble:.1%}")
pbks_prob = ensemble

print("\nContextual Adjustments (positive = favor PBKS):")
adjustments = 0
print("  + PBKS undefeated at Mullanpur home in 2026 - perfect home record: +6%")
adjustments += 0.06
print("  + PBKS must-win desperation - 5 losses in a row, playoff hopes shrinking: +3%")
adjustments += 0.03
print("  + Shreyas Iyer top form (396 runs, avg 49.50) - PBKS captain anchoring middle: +3%")
adjustments += 0.03
print("  + Cooper Connolly 377+ runs - PBKS' leading run-scorer: +2%")
adjustments += 0.02
print("  + Arshdeep Singh new-ball threat at Mullanpur - early carry for pacers: +2%")
adjustments += 0.02
print("  + Afternoon 3:30 PM start = no dew advantage for RCB chase: +2%")
adjustments += 0.02
print("  - RCB top of table (16 pts), peak form - 4 wins in last 5: -7%")
adjustments -= 0.07
print("  - Virat Kohli purple-hot: 387 runs, avg 48, SR 164 + 105* vs KKR last game: -5%")
adjustments -= 0.05
print("  - Bhuvneshwar Kumar Purple Cap leader (21 wickets at econ 7.46): -3%")
adjustments -= 0.03
print("  - Rajat Patidar SR 195 with 3 fifties - aggressive captain firing: -3%")
adjustments -= 0.03
print("  - RCB beat PBKS in 2025 final - psychological edge over Punjab: -2%")
adjustments -= 0.02
print("  - PBKS lost 5 in a row - dressing room confidence shattered: -3%")
adjustments -= 0.03
print("  - RCB chasing brilliance in 2026 - flat Mullanpur pitch suits big-totals/chase: -2%")
adjustments -= 0.02

final_pbks = max(0.10, min(0.90, pbks_prob + adjustments))
final_rcb = 1.0 - final_pbks
print(f"\nFinal P(PBKS wins): {final_pbks:.1%}")
print(f"Final P(RCB wins): {final_rcb:.1%}")

np.random.seed(42)
pbks_wins = final_pbks >= 0.5
winner_abbrev = 'PBKS' if pbks_wins else 'RCB'
winner_full = 'Punjab Kings' if pbks_wins else 'Royal Challengers Bangalore'
winner_conf = int(final_pbks * 100) if pbks_wins else int(final_rcb * 100)

# Mullanpur: par 175-190, batting friendly, totals frequently 200+
if pbks_wins:
    pbks_score_mu, rcb_score_mu = 190, 173
else:
    pbks_score_mu, rcb_score_mu = 170, 195
pbks_score = np.random.normal(loc=pbks_score_mu, scale=18, size=10000)
rcb_score = np.random.normal(loc=rcb_score_mu, scale=18, size=10000)
if pbks_wins:
    margin = pbks_score - rcb_score
else:
    margin = rcb_score - pbks_score
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
