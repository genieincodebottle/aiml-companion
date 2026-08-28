"""
Chennai Super Kings vs Sunrisers Hyderabad - Match 63, IPL 2026 - May 18 (7:30 PM IST)
ML Prediction Script using Ensemble Models
Venue: MA Chidambaram Stadium, Chennai (CSK home, Chepauk)
Context:
  CSK 11P 6W-5L = 12 pts (8th but games in hand). Recent: huge 103-run win over MI; lost to GT by 8w.
  SRH 12P 7W-5L = 14 pts (3rd, in playoff hunt). Form: beat CSK (M27 by 10 runs), beat DC, beat RR, beat MI - solid run.
  H2H this season: SRH won M27 in Hyderabad by 10 runs.
  Stakes: CSK MUST WIN to keep top-4 hopes; SRH near-confirmed for playoffs, can play freely.
  Chepauk: traditionally spin-friendly, slows in 2nd innings, par 165-180. CSK rebuilt squad - Conway, Ruturaj, Dhoni, Jadeja, Pathirana, Theekshana.
  SRH firepower: Travis Head, Abhishek Sharma, Klaasen, Cummins, Bhuvneshwar.
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
    # Updated Elo as of May 18 morning (post-M62)
    elo_ratings = {
        'Royal Challengers Bangalore': 1830,
        'Gujarat Titans': 1815,
        'Sunrisers Hyderabad': 1745,   # 7W-5L, beat CSK + MI + DC + RR recently, top-4 lock
        'Chennai Super Kings': 1710,   # 6W-5L, at Chepauk strong but recent mixed
        'Punjab Kings': 1685,
        'Delhi Capitals': 1665,
        'Rajasthan Royals': 1635,
        'Kolkata Knight Riders': 1640,
        'Mumbai Indians': 1500,
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
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.62
    else:
        features['venue_home_win_pct'] = 0.62

    features['home_advantage'] = 1.10  # Chepauk strong home advantage for CSK
    features['toss_factor'] = 1.05
    # Chepauk evening: dew minor, ball grips, spin friendly, bat-first slight edge but chasers win often
    features['chase_factor'] = 1.02

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.85,
        'Gujarat Titans': 0.80,
        'Sunrisers Hyderabad': 0.72,    # 4 of last 5 wins, top form
        'Chennai Super Kings': 0.62,    # Mixed: thrashed MI but lost to GT
        'Punjab Kings': 0.20,
        'Mumbai Indians': 0.45,
        'Delhi Capitals': 0.48,         # beat RR last out
        'Rajasthan Royals': 0.28,
        'Kolkata Knight Riders': 0.50,
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Chennai Super Kings'
away_team = 'Sunrisers Hyderabad'
venue = 'Chidambaram'

print(f"\n{'='*60}")
print(f"MATCH 63: {home_team} vs {away_team}")
print(f"Venue: MA Chidambaram Stadium, Chennai (Chepauk)")
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
    print(f"  {name.upper()} P(CSK wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(CSK wins): {ensemble:.1%}")
csk_prob = ensemble

print("\nContextual Adjustments (positive = favor CSK):")
adjustments = 0
print("  + CSK home at Chepauk - historically 60%+ home win rate, spin-friendly: +6%")
adjustments += 0.06
print("  + Must-win for CSK playoff math (12 pts, NRR boost needed): +3%")
adjustments += 0.03
print("  + Jadeja + Theekshana spin twins decisive on Chepauk surface: +3%")
adjustments += 0.03
print("  + Pathirana yorker death-overs threat vs SRH's middle order: +2%")
adjustments += 0.02
print("  + Dhoni finisher in late overs at home crowd: +2%")
adjustments += 0.02
print("  + Conway-Ruturaj top order back to form in M44/M62 wins: +2%")
adjustments += 0.02
print("  - SRH won the first H2H M27 by 10 runs (head-to-head 2026 edge): -4%")
adjustments -= 0.04
print("  - SRH on hot streak (won 4 of last 5) vs CSK mixed form: -3%")
adjustments -= 0.03
print("  - Travis Head + Abhishek Sharma powerplay carnage hard to contain: -3%")
adjustments -= 0.03
print("  - Klaasen + Nitish Reddy middle-order can clear 200 even on slower decks: -3%")
adjustments -= 0.03
print("  - Pat Cummins captaincy + new-ball spell can break CSK top order: -2%")
adjustments -= 0.02
print("  - SRH near-confirmed for playoffs - low pressure, free-flowing cricket: -2%")
adjustments -= 0.02

final_csk = max(0.10, min(0.90, csk_prob + adjustments))
final_srh = 1.0 - final_csk
print(f"\nFinal P(CSK wins): {final_csk:.1%}")
print(f"Final P(SRH wins): {final_srh:.1%}")

np.random.seed(42)
csk_wins = final_csk >= 0.5
winner_abbrev = 'CSK' if csk_wins else 'SRH'
winner_full = 'Chennai Super Kings' if csk_wins else 'Sunrisers Hyderabad'
winner_conf = int(final_csk * 100) if csk_wins else int(final_srh * 100)

# Chepauk: par 165-180, slows in 2nd innings, spin friendly
if csk_wins:
    csk_score_mu, srh_score_mu = 178, 162
else:
    csk_score_mu, srh_score_mu = 165, 188
csk_score = np.random.normal(loc=csk_score_mu, scale=18, size=10000)
srh_score = np.random.normal(loc=srh_score_mu, scale=18, size=10000)
if csk_wins:
    margin = csk_score - srh_score
else:
    margin = srh_score - csk_score
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
