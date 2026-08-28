"""
Delhi Capitals vs Rajasthan Royals - Match 62, IPL 2026 - May 17 (7:30 PM)
ML Prediction Script using Ensemble Models
Venue: Arun Jaitley Stadium, Delhi (DC home)
Context: Both teams nearly out of playoff race. DC (7th, 10 pts, 5W-7L, -0.99 NRR) vs RR (5th, struggling, lost recently to GT). Spoiler match with playoff dignity at stake.
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
    # Updated Elo as of May 16 evening
    elo_ratings = {
        'Royal Challengers Bangalore': 1830,
        'Gujarat Titans': 1815,
        'Mumbai Indians': 1500,
        'Punjab Kings': 1685,
        'Rajasthan Royals': 1635,                     # 5th, recent loss to GT, inconsistent
        'Chennai Super Kings': 1700,
        'Sunrisers Hyderabad': 1720,
        'Delhi Capitals': 1660,                       # beat PBKS, but 5W-7L overall
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

    features['home_advantage'] = 1.05  # Arun Jaitley moderate home advantage
    features['toss_factor'] = 1.05
    # Arun Jaitley relaid surface: bat first preferred (7 of 10 wins), but dew evening complex
    # 7:30 PM start = evening dew minor in May but slows pitch in 2nd innings
    features['chase_factor'] = 0.92  # bat first slightly favored

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.85,
        'Gujarat Titans': 0.80,
        'Mumbai Indians': 0.55,
        'Punjab Kings': 0.20,
        'Rajasthan Royals': 0.32,                     # 1 win in last 5, sliding
        'Chennai Super Kings': 0.70,
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.45,                       # beat PBKS, but 7 losses overall
        'Kolkata Knight Riders': 0.50,
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Delhi Capitals'
away_team = 'Rajasthan Royals'
venue = 'Arun Jaitley'

print(f"\n{'='*60}")
print(f"MATCH 62: {home_team} vs {away_team}")
print(f"Venue: Arun Jaitley Stadium, Delhi")
print(f"Time: 7:30 PM IST (evening)")
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
print("  + DC home at Arun Jaitley - familiar conditions, home crowd: +5%")
adjustments += 0.05
print("  + KL Rahul anchoring DC top order - experienced in Delhi conditions: +3%")
adjustments += 0.03
print("  + Axar Patel (c) all-round contribution + David Miller finisher: +2%")
adjustments += 0.02
print("  + Mitchell Starc + Mukesh Kumar pace combo suits Arun Jaitley early swing: +3%")
adjustments += 0.03
print("  + DC must-win for playoff math - 50+ margin needed for NRR boost: +2%")
adjustments += 0.02
print("  + Arun Jaitley bat-first edge (7 of 10 since relay) suits DC if they win toss: +2%")
adjustments += 0.02
print("  + Tristan Stubbs middle-order power - dangerous on flat Delhi surface: +2%")
adjustments += 0.02
print("  - DC -0.99 NRR signals heavy past defeats, structural weakness: -3%")
adjustments -= 0.03
print("  - DC 5W-7L from 12 vs RR's better top-3 talent depth: -2%")
adjustments -= 0.02
print("  - Jaiswal-Samson top-order match-winners on flat decks: -3%")
adjustments -= 0.03
print("  - Jofra Archer pace + Maheesh Theekshana mystery spin attack: -3%")
adjustments -= 0.03
print("  - Wanindu Hasaranga wrist-spin lethal under Delhi lights: -2%")
adjustments -= 0.02
print("  - DC all-time H2H lead is razor-thin (16-15) - very even rivalry: -1%")
adjustments -= 0.01

final_dc = max(0.10, min(0.90, dc_prob + adjustments))
final_rr = 1.0 - final_dc
print(f"\nFinal P(DC wins): {final_dc:.1%}")
print(f"Final P(RR wins): {final_rr:.1%}")

np.random.seed(42)
dc_wins = final_dc >= 0.5
winner_abbrev = 'DC' if dc_wins else 'RR'
winner_full = 'Delhi Capitals' if dc_wins else 'Rajasthan Royals'
winner_conf = int(final_dc * 100) if dc_wins else int(final_rr * 100)

# Arun Jaitley: par 180-200, batting friendly, relaid surface
if dc_wins:
    dc_score_mu, rr_score_mu = 190, 174
else:
    dc_score_mu, rr_score_mu = 172, 195
dc_score = np.random.normal(loc=dc_score_mu, scale=18, size=10000)
rr_score = np.random.normal(loc=rr_score_mu, scale=18, size=10000)
if dc_wins:
    margin = dc_score - rr_score
else:
    margin = rr_score - dc_score
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
