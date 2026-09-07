"""
Punjab Kings vs Mumbai Indians - Match 58, IPL 2026 - May 14
ML Prediction Script using Ensemble Models
Venue: HPCA Stadium, Dharamsala (PBKS secondary home)
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
    # Updated Elo as of May 14 (post matches 56-57)
    # PBKS: 13 pts (#4), 6W-5L, NRR +0.428, 4-match losing streak, lost at Dharamsala vs DC
    # MI: ELIMINATED (#9), 3W-8L, 6 pts/11, lost final ball thriller vs RCB at Raipur
    elo_ratings = {
        'Royal Challengers Bangalore': 1790,
        'Gujarat Titans': 1770,
        'Mumbai Indians': 1480,                # eliminated but Rohit in vintage form
        'Punjab Kings': 1735,                  # slid from #2 to #4 on 4-match losing streak
        'Rajasthan Royals': 1660,
        'Chennai Super Kings': 1640,
        'Sunrisers Hyderabad': 1765,
        'Delhi Capitals': 1655,
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

    features['home_advantage'] = 0.95
    features['toss_factor'] = 1.05
    # Dharamsala HPCA: small square boundaries 63/65m, recent games 200+ regularly
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.72,
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.40,                # 3W-8L overall, 2-3 last 5 incl narrow loss vs RCB
        'Punjab Kings': 0.30,                  # 4 successive losses but 6-5 season overall
        'Rajasthan Royals': 0.35,
        'Chennai Super Kings': 0.55,
        'Sunrisers Hyderabad': 0.65,
        'Delhi Capitals': 0.60,
        'Kolkata Knight Riders': 0.65,
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Punjab Kings'
away_team = 'Mumbai Indians'
venue = 'Himachal Pradesh Cricket Association Stadium'

print(f"\n{'='*60}")
print(f"MATCH 58: {home_team} vs {away_team}")
print(f"Venue: HPCA Stadium, Dharamsala")
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
print("  + PBKS fundamentally stronger this season: 6W-5L vs MI 3W-8L: +8%")
adjustments += 0.08
print("  + PBKS still in playoff race (13 pts/11) vs MI eliminated (6 pts/11): +4%")
adjustments += 0.04
print("  + Shreyas Iyer in red-hot form: 3 successive 50s incl 59*(40) vs DC: +3%")
adjustments += 0.03
print("  + Priyansh Arya strike rate 226 leads PBKS top order: +2%")
adjustments += 0.02
print("  + PBKS nominal home at Dharamsala, must-win for playoffs: +2%")
adjustments += 0.02
print("  + Chahal will likely bowl this match - leg-spin threat vs SKY/Tilak: +2%")
adjustments += 0.02
print("  - PBKS on 4-match losing streak including last home game at Dharamsala vs DC: -5%")
adjustments -= 0.05
print("  - Arshdeep Singh under pressure (11W/10M, 10.5 econ) - leaks runs: -2%")
adjustments -= 0.02
print("  - Rohit Sharma 84(44) last out - back to vintage form, dangerous: -3%")
adjustments -= 0.03
print("  - Tilak Varma 57(43) last match - middle order finding rhythm: -2%")
adjustments -= 0.02
print("  - MI playing free of pressure (eliminated), Hardik back from spasm: -2%")
adjustments -= 0.02
print("  - Bumrah even out of form still elite at death - PBKS 200+ scores capped: -2%")
adjustments -= 0.02
print("  - Dharamsala small square boundaries (63/65m) suit MI power hitters: -2%")
adjustments -= 0.02

final_pbks = max(0.10, min(0.90, pbks_prob + adjustments))
final_mi = 1.0 - final_pbks
print(f"\nFinal P(PBKS wins): {final_pbks:.1%}")
print(f"Final P(MI wins): {final_mi:.1%}")

np.random.seed(42)
pbks_wins = final_pbks >= 0.5
winner_abbrev = 'PBKS' if pbks_wins else 'MI'
winner_full = 'Punjab Kings' if pbks_wins else 'Mumbai Indians'
winner_conf = int(final_pbks * 100) if pbks_wins else int(final_mi * 100)

# Dharamsala: par ~175-185, batting-friendly, small boundaries
if pbks_wins:
    pbks_score_mu, mi_score_mu = 188, 170
else:
    pbks_score_mu, mi_score_mu = 170, 188
pbks_score = np.random.normal(loc=pbks_score_mu, scale=20, size=10000)
mi_score = np.random.normal(loc=mi_score_mu, scale=20, size=10000)
if pbks_wins:
    margin = pbks_score - mi_score
else:
    margin = mi_score - pbks_score
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
