"""
Lucknow Super Giants vs Royal Challengers Bengaluru - Match 50, IPL 2026 - May 7
ML Prediction Script using Ensemble Models
Venue: BRSABV Ekana Cricket Stadium, Lucknow
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
    # Updated Elo as of May 7 (after 49 matches)
    # LSG: 4 pts, 10th place, 2W-7L, on 6-match losing streak (DESPERATE)
    # RCB: 12 pts, top 4 (defending champs), 6W-3L, NRR +1.420, firm playoff favorites
    elo_ratings = {
        'Royal Challengers Bangalore': 1755,    # defending champs, 6W-3L, Kohli orange cap
        'Gujarat Titans': 1730,
        'Mumbai Indians': 1495,
        'Punjab Kings': 1770,
        'Rajasthan Royals': 1700,
        'Chennai Super Kings': 1640,
        'Sunrisers Hyderabad': 1755,
        'Delhi Capitals': 1685,
        'Kolkata Knight Riders': 1520,
        'Lucknow Super Giants': 1450,           # 2W-7L, six straight losses, near-eliminated
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
    # Ekana: spin-friendly, par 165-175, dew helps chasing in evening matches
    features['chase_factor'] = 1.08

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.70,    # 6W-3L, defending champs, top-4 lock
        'Gujarat Titans': 0.70,
        'Mumbai Indians': 0.25,
        'Punjab Kings': 0.50,
        'Rajasthan Royals': 0.40,
        'Chennai Super Kings': 0.40,
        'Sunrisers Hyderabad': 0.85,
        'Delhi Capitals': 0.55,
        'Kolkata Knight Riders': 0.45,
        'Lucknow Super Giants': 0.10,           # six straight losses, virtually out
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Lucknow Super Giants'
away_team = 'Royal Challengers Bangalore'
venue = 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium'

print(f"\n{'='*60}")
print(f"MATCH 50: {home_team} vs {away_team}")
print(f"Venue: BRSABV Ekana Cricket Stadium, Lucknow")
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
    print(f"  {name.upper()} P(LSG wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(LSG wins): {ensemble:.1%}")
lsg_prob = ensemble

print("\nContextual Adjustments (positive = favor LSG):")
adjustments = 0
print("  + Home advantage at Ekana, spin-friendly track LSG bowlers know: +4%")
adjustments += 0.04
print("  + LSG won 2 of 3 home matches vs RCB (Pant 118* in 2025): +3%")
adjustments += 0.03
print("  + LSG must-win desperation, mathematically alive (4-of-5 needed): +2%")
adjustments += 0.02
print("  + Spin pitch suits Hasaranga / Bishnoi-style attack vs RCB middle: +2%")
adjustments += 0.02
print("  + RCB Josh Hazlewood reportedly OUT (9 wkts in 3 H2H, eco 7.54): +3%")
adjustments += 0.03
print("  - RCB on 6W-3L, top-4 lock, defending champs in form: -7%")
adjustments -= 0.07
print("  - Virat Kohli Orange Cap leader (379 runs), peak form: -5%")
adjustments -= 0.05
print("  - LSG on 6-match losing streak, batting brittle (Pooran 82@10 avg): -6%")
adjustments -= 0.06
print("  - Rajat Patidar (257) + Tim David finishing power vs LSG bowlers: -3%")
adjustments -= 0.03
print("  - H2H: RCB leads 5-2 overall, won most recent fixture (Match 23): -3%")
adjustments -= 0.03
print("  - LSG fail to defend totals (228 vs MI lost) - bowling concerns: -3%")
adjustments -= 0.03
print("  - Krunal Pandya all-rounder thrives on spin tracks for RCB: -2%")
adjustments -= 0.02

final_lsg = max(0.10, min(0.90, lsg_prob + adjustments))
final_rcb = 1.0 - final_lsg
print(f"\nFinal P(LSG wins): {final_lsg:.1%}")
print(f"Final P(RCB wins): {final_rcb:.1%}")

np.random.seed(42)
lsg_wins = final_lsg >= 0.5
winner_abbrev = 'LSG' if lsg_wins else 'RCB'
winner_full = 'Lucknow Super Giants' if lsg_wins else 'Royal Challengers Bengaluru'
winner_conf = int(final_lsg * 100) if lsg_wins else int(final_rcb * 100)

# Ekana scoring: par 165-175, RCB chases big totals well
if lsg_wins:
    lsg_score_mu, rcb_score_mu = 178, 158
else:
    lsg_score_mu, rcb_score_mu = 155, 180
lsg_score = np.random.normal(loc=lsg_score_mu, scale=20, size=10000)
rcb_score = np.random.normal(loc=rcb_score_mu, scale=20, size=10000)
if lsg_wins:
    margin = lsg_score - rcb_score
else:
    margin = rcb_score - lsg_score
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
