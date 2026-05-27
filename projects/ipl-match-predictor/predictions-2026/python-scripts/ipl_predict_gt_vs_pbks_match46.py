"""
Gujarat Titans vs Punjab Kings - Match 46, IPL 2026 - May 3
ML Prediction Script using Ensemble Models (PRE-MATCH PERSPECTIVE - BACKFILL)
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
    # Updated Elo based on IPL 2026 standings as of May 3 (PRE-MATCH 46)
    # PBKS 13pts top, GT 12pts 5th (2 successive wins, surging)
    elo_ratings = {
        'Royal Challengers Bangalore': 1745,
        'Gujarat Titans': 1715,           # 5W-3L, 12 pts, 2 wins on trot
        'Mumbai Indians': 1500,
        'Punjab Kings': 1800,             # table-topper, 6W-2L, 13 pts
        'Rajasthan Royals': 1715,
        'Chennai Super Kings': 1620,
        'Sunrisers Hyderabad': 1755,
        'Delhi Capitals': 1665,
        'Kolkata Knight Riders': 1490,
        'Lucknow Super Giants': 1485,
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
    # Narendra Modi Stadium Ahmedabad: large boundaries, balanced wicket, dew at night helps chase
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.70,           # 2 successive wins, on the rise
        'Mumbai Indians': 0.20,
        'Punjab Kings': 0.90,             # table-topper, peak form
        'Rajasthan Royals': 0.55,
        'Chennai Super Kings': 0.40,
        'Sunrisers Hyderabad': 0.85,
        'Delhi Capitals': 0.55,
        'Kolkata Knight Riders': 0.30,
        'Lucknow Super Giants': 0.15,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Gujarat Titans'
away_team = 'Punjab Kings'
venue = 'Narendra Modi'

print(f"\n{'='*60}")
print(f"MATCH 46: {home_team} vs {away_team}")
print(f"Venue: Narendra Modi Stadium, Ahmedabad, Date: May 3, 2026")
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
print("  + GT home at Narendra Modi Stadium - largest crowd in IPL, huge home advantage: +5%")
adjustments += 0.05
print("  + GT on 2-match winning streak, momentum + balanced squad: +4%")
adjustments += 0.04
print("  + Sai Sudharsan in fine form, Shubman Gill anchor, Buttler firepower: +3%")
adjustments += 0.03
print("  + Rashid Khan + Mohit Sharma a fearsome death-bowling combination: +3%")
adjustments += 0.03
print("  + GT's all-rounders (Washington, Tewatia, Holder) suit Ahmedabad's larger boundaries: +2%")
adjustments += 0.02
print("  - PBKS table-toppers (13 pts, 6W-2L), most consistent team this season: -6%")
adjustments -= 0.06
print("  - PBKS won both prior visits to Ahmedabad vs GT, venue mental edge: -4%")
adjustments -= 0.04
print("  - Shreyas Iyer in elite form leading PBKS, Yuzvendra Chahal/Arshdeep deadly: -3%")
adjustments -= 0.03
print("  - PBKS depth and bench strength - fewer weak links than GT this year: -2%")
adjustments -= 0.02

final_gt = max(0.10, min(0.90, gt_prob + adjustments))
final_pbks = 1.0 - final_gt
print(f"\nFinal P(GT wins):   {final_gt:.1%}")
print(f"Final P(PBKS wins):  {final_pbks:.1%}")

np.random.seed(42)
gt_wins = final_gt >= 0.5
winner_abbrev = 'GT' if gt_wins else 'PBKS'
winner_full = 'Gujarat Titans' if gt_wins else 'Punjab Kings'
winner_conf = int(final_gt * 100) if gt_wins else int(final_pbks * 100)

if gt_wins:
    gt_score_mu, pbks_score_mu = 178, 162
else:
    gt_score_mu, pbks_score_mu = 165, 182
gt_score = np.random.normal(loc=gt_score_mu, scale=20, size=10000)
pbks_score = np.random.normal(loc=pbks_score_mu, scale=22, size=10000)
if gt_wins:
    margin = gt_score - pbks_score
else:
    margin = pbks_score - gt_score
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
