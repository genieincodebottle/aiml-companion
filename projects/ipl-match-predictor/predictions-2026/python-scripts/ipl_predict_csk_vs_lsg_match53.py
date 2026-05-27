"""
Chennai Super Kings vs Lucknow Super Giants - Match 53, IPL 2026 - May 10
ML Prediction Script using Ensemble Models
Venue: MA Chidambaram Stadium, Chennai (3:30 PM IST - afternoon, no dew)
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
    # Updated Elo as of May 9 (after 51 matches)
    # CSK: 10 pts, 6th, 5W-5L, won 2 in a row (beat MI by 103 runs, beat DC by 8 wkts).
    #      Sanju Samson now at CSK with 400+ runs.
    # LSG: 6 pts, 10th, 3W-7L. Just beat RCB by 9 runs (snapped own losing streak).
    #      Pant captain. Prince Yadav 16 wkts.
    elo_ratings = {
        'Royal Challengers Bangalore': 1755,
        'Gujarat Titans': 1745,
        'Mumbai Indians': 1505,
        'Punjab Kings': 1770,
        'Rajasthan Royals': 1685,
        'Chennai Super Kings': 1685,        # bump for 2 in a row + Samson signing
        'Sunrisers Hyderabad': 1740,
        'Delhi Capitals': 1640,
        'Kolkata Knight Riders': 1590,
        'Lucknow Super Giants': 1465,       # tiny bump after RCB win, but still bottom
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
    features['toss_factor'] = 1.02
    # Chepauk afternoon (3:30pm IST): NO dew, slight bat-first preference
    # Pitch turning, spinners get help. CSK has Jadeja + Ashwin (if playing).
    features['chase_factor'] = 0.95   # afternoon, bat-first slight edge

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.40,
        'Punjab Kings': 0.55,
        'Rajasthan Royals': 0.35,
        'Chennai Super Kings': 0.65,        # 2 wins on the trot
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.25,
        'Kolkata Knight Riders': 0.65,
        'Lucknow Super Giants': 0.30,       # 3W-7L, but just beat RCB
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Chennai Super Kings'
away_team = 'Lucknow Super Giants'
venue = 'MA Chidambaram Stadium'

print(f"\n{'='*60}")
print(f"MATCH 53: {home_team} vs {away_team}")
print(f"Venue: MA Chidambaram Stadium, Chennai (afternoon, 3:30 PM IST)")
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
print("  + CSK home at Chepauk (fortress): historical home_win ~64% in dataset: +6%")
adjustments += 0.06
print("  + CSK on 2-match winning streak (103-run win vs MI, 8-wicket win vs DC): +5%")
adjustments += 0.05
print("  + Sanju Samson at CSK with 400+ runs and two centuries this season: +4%")
adjustments += 0.04
print("  + Ruturaj Gaikwad 251 runs incl 2 fifties, captaincy on point: +2%")
adjustments += 0.02
print("  + Jadeja-led spin attack on slow turning Chepauk pitch (afternoon): +5%")
adjustments += 0.05
print("  + Afternoon match - no dew, bat-first edge favours CSK first innings strategy: +2%")
adjustments += 0.02
print("  - LSG just beat RCB by 9 runs, snapped losing streak, confidence boost: -3%")
adjustments -= 0.03
print("  - H2H last 6 meetings: LSG 3, CSK 2, NR 1 (LSG slight historical edge): -3%")
adjustments -= 0.03
print("  - Prince Yadav 16 wkts, top LSG bowler can hurt CSK middle overs: -2%")
adjustments -= 0.02
print("  - Rishabh Pant captain - X-factor batter in pressure situations: -2%")
adjustments -= 0.02

final_csk = max(0.10, min(0.90, csk_prob + adjustments))
final_lsg = 1.0 - final_csk
print(f"\nFinal P(CSK wins): {final_csk:.1%}")
print(f"Final P(LSG wins): {final_lsg:.1%}")

np.random.seed(42)
csk_wins = final_csk >= 0.5
winner_abbrev = 'CSK' if csk_wins else 'LSG'
winner_full = 'Chennai Super Kings' if csk_wins else 'Lucknow Super Giants'
winner_conf = int(final_csk * 100) if csk_wins else int(final_lsg * 100)

# Chepauk afternoon: par ~165-175, spinners help, lower scoring
if csk_wins:
    csk_score_mu, lsg_score_mu = 178, 158
else:
    csk_score_mu, lsg_score_mu = 158, 178
csk_score = np.random.normal(loc=csk_score_mu, scale=20, size=10000)
lsg_score = np.random.normal(loc=lsg_score_mu, scale=20, size=10000)
if csk_wins:
    margin = csk_score - lsg_score
else:
    margin = lsg_score - csk_score
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
