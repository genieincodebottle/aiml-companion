"""
Rajasthan Royals vs Lucknow Super Giants - Match 64, IPL 2026 - May 19 (7:30 PM IST)
ML Prediction Script using Ensemble Models
Venue: Barsapara Cricket Stadium, Guwahati (RR secondary home)
Context:
  RR 12P 6W-6L = 12 pts (mathematically alive but NRR sunk by GT 77-run loss + DC defeat).
  LSG 11P 3W-8L = 6 pts (eliminated, playing for pride only).
  H2H this season: RR won M32 in Lucknow by 40 runs.
  Recent form:
    RR last 5: L(DC-W5), L(GT-77), L(DC), W(PBKS), L(SRH) = 1W-4L cold.
    LSG last 5: L(CSK), W(RCB upset), L(MI), L(KKR super-over), L(RR-40) = 1W-4L cold.
  Both teams cold but RR has class players: Jaiswal, Samson, Archer, Hasaranga, Theekshana.
  LSG has lost top talent (KL Rahul gone to DC, Pooran in patchy form), Marcus Stoinis injury risk.
  Guwahati Barsapara: par 175-185, evening dew moderate, slight chase edge but bat-first viable.
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
    # Updated Elo as of May 18 evening (post-M62)
    elo_ratings = {
        'Royal Challengers Bangalore': 1830,
        'Gujarat Titans': 1815,
        'Sunrisers Hyderabad': 1745,
        'Chennai Super Kings': 1710,
        'Punjab Kings': 1685,
        'Delhi Capitals': 1665,
        'Rajasthan Royals': 1620,        # slipped after GT 77-run + DC loss
        'Kolkata Knight Riders': 1640,
        'Mumbai Indians': 1500,
        'Lucknow Super Giants': 1420,    # bottom of table, eliminated
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

    features['home_advantage'] = 1.04   # Barsapara is RR's secondary home, modest edge
    features['toss_factor'] = 1.05
    features['chase_factor'] = 1.04     # slight chase edge under evening lights

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.85,
        'Gujarat Titans': 0.80,
        'Sunrisers Hyderabad': 0.72,
        'Chennai Super Kings': 0.65,
        'Punjab Kings': 0.20,
        'Mumbai Indians': 0.45,
        'Delhi Capitals': 0.52,
        'Rajasthan Royals': 0.22,         # 1W-4L last 5, cold
        'Kolkata Knight Riders': 0.50,
        'Lucknow Super Giants': 0.25,     # 1W-4L last 5, also cold
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Rajasthan Royals'
away_team = 'Lucknow Super Giants'
venue = 'Barsapara'

print(f"\n{'='*60}")
print(f"MATCH 64: {home_team} vs {away_team}")
print(f"Venue: Barsapara Cricket Stadium, Guwahati")
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
    print(f"  {name.upper()} P(RR wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(RR wins): {ensemble:.1%}")
rr_prob = ensemble

print("\nContextual Adjustments (positive = favor RR):")
adjustments = 0
print("  + RR won H2H M32 by 40 runs - significant intent advantage: +5%")
adjustments += 0.05
print("  + Barsapara is RR secondary home - familiar surface, home support: +3%")
adjustments += 0.03
print("  + Jaiswal-Samson top order class differential vs LSG bowling: +4%")
adjustments += 0.04
print("  + Jofra Archer + Hasaranga + Theekshana attack outclasses LSG: +3%")
adjustments += 0.03
print("  + LSG eliminated - mental drift risk, lower stakes for them: +2%")
adjustments += 0.02
print("  + RR mathematically alive (slim) - more competitive intent: +2%")
adjustments += 0.02
print("  - Both teams ice-cold form (1W-4L last 5) - flip risk: -2%")
adjustments -= 0.02
print("  - RR humiliated by GT (77 runs) and lost to DC - psychological damage: -3%")
adjustments -= 0.03
print("  - LSG upset RCB at home (9 runs) showing they can spike: -2%")
adjustments -= 0.02
print("  - Nicholas Pooran can demolish any attack on his day: -2%")
adjustments -= 0.02
print("  - Avesh Khan + Mohsin Khan have pace + control vs RR top order: -1%")
adjustments -= 0.01

final_rr = max(0.10, min(0.90, rr_prob + adjustments))
final_lsg = 1.0 - final_rr
print(f"\nFinal P(RR wins): {final_rr:.1%}")
print(f"Final P(LSG wins): {final_lsg:.1%}")

np.random.seed(42)
rr_wins = final_rr >= 0.5
winner_abbrev = 'RR' if rr_wins else 'LSG'
winner_full = 'Rajasthan Royals' if rr_wins else 'Lucknow Super Giants'
winner_conf = int(final_rr * 100) if rr_wins else int(final_lsg * 100)

# Barsapara: par 175-185, batting friendly, slight chase edge
if rr_wins:
    rr_score_mu, lsg_score_mu = 188, 168
else:
    rr_score_mu, lsg_score_mu = 165, 192
rr_score = np.random.normal(loc=rr_score_mu, scale=18, size=10000)
lsg_score = np.random.normal(loc=lsg_score_mu, scale=18, size=10000)
if rr_wins:
    margin = rr_score - lsg_score
else:
    margin = lsg_score - rr_score
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
