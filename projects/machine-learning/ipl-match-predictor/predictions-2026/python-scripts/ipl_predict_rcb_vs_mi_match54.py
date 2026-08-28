"""
Royal Challengers Bengaluru vs Mumbai Indians - Match 54, IPL 2026 - May 10
ML Prediction Script using Ensemble Models
Venue: Shaheed Veer Narayan Singh Stadium, Raipur (RCB home, away from Bengaluru)
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
    # RCB: 12 pts, 3rd, 6W-4L. Just lost to LSG by 9 runs (DLS chase 213). Patidar capt + Andy Flower.
    #      Kohli, Tim David, Salt batting; Hazlewood, Bhuvneshwar, Thushara bowling.
    # MI: 6 pts, 9th, 3W-7L. Just beat LSG (chase 229) by 6 wkts. Hardik back from spasm.
    #     Suryakumar 195 runs avg 19.5 (struggling); Rohit 84 off 44 last match.
    elo_ratings = {
        'Royal Challengers Bangalore': 1750,    # tiny dip after LSG loss but still strong
        'Gujarat Titans': 1745,
        'Mumbai Indians': 1530,                 # bump for win + Hardik return
        'Punjab Kings': 1770,
        'Rajasthan Royals': 1685,
        'Chennai Super Kings': 1685,
        'Sunrisers Hyderabad': 1740,
        'Delhi Capitals': 1640,
        'Kolkata Knight Riders': 1590,
        'Lucknow Super Giants': 1465,
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

    # NOTE: Raipur not in historical CSV (rare venue). Use 0.50 as neutral.
    venue_matches = historical_df[historical_df['venue'].str.contains(venue, na=False, case=False)]
    if len(venue_matches) > 0:
        home_wins = len(venue_matches[(venue_matches['team1'] == home_team) & (venue_matches['winner'] == home_team)])
        home_total = len(venue_matches[venue_matches['team1'] == home_team])
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.50
    else:
        features['venue_home_win_pct'] = 0.50    # neutral - not RCB's true home

    # Reduced home advantage since Raipur is not Chinnaswamy (still RCB home crowd, dressing room)
    features['home_advantage'] = 0.85
    features['toss_factor'] = 1.05
    # Raipur evening: dew helps batting in 2nd innings, slippery ball
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.55,    # 6W-4L but lost last
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.45,                 # 3W-7L but won 2 of last 3
        'Punjab Kings': 0.55,
        'Rajasthan Royals': 0.35,
        'Chennai Super Kings': 0.65,
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.25,
        'Kolkata Knight Riders': 0.65,
        'Lucknow Super Giants': 0.30,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Royal Challengers Bangalore'
away_team = 'Mumbai Indians'
venue = 'Shaheed Veer Narayan Singh Stadium'

print(f"\n{'='*60}")
print(f"MATCH 54: {home_team} vs {away_team}")
print(f"Venue: Shaheed Veer Narayan Singh Stadium, Raipur (RCB home, away from Bengaluru)")
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
print("  + RCB stronger overall position: 12 pts vs MI's 6, 3rd vs 9th in standings: +6%")
adjustments += 0.06
print("  + Kohli, Patidar, Tim David, Salt - deep batting lineup: +4%")
adjustments += 0.04
print("  + Hazlewood, Bhuvneshwar, Thushara give RCB strong pace attack: +3%")
adjustments += 0.03
print("  + RCB had pulled to 2nd recently with 6 wins in 8 matches: +3%")
adjustments += 0.03
print("  + Patidar's RCB still has best NRR among 12-pt teams (+1.234): +2%")
adjustments += 0.02
print("  - Raipur is RCB home in name only, not Chinnaswamy - reduced home advantage: -3%")
adjustments -= 0.03
print("  - MI just beat LSG chasing 229 - chase confidence high, dew at Raipur helps: -3%")
adjustments -= 0.03
print("  - Hardik Pandya returning from back spasm gives MI better balance: -2%")
adjustments -= 0.02
print("  - Rohit Sharma 84 off 44 last match - in form opener: -2%")
adjustments -= 0.02
print("  - MI have Bumrah - best death bowler vs Patidar/Tim David death overs: -3%")
adjustments -= 0.03
print("  - RCB lost last match (LSG by 9 runs DLS), momentum dented: -1%")
adjustments -= 0.01
print("  - H2H all-time: MI leads 19-16 (54%) across 35 IPL meetings: -1%")
adjustments -= 0.01

final_rcb = max(0.10, min(0.90, rcb_prob + adjustments))
final_mi = 1.0 - final_rcb
print(f"\nFinal P(RCB wins): {final_rcb:.1%}")
print(f"Final P(MI wins): {final_mi:.1%}")

np.random.seed(42)
rcb_wins = final_rcb >= 0.5
winner_abbrev = 'RCB' if rcb_wins else 'MI'
winner_full = 'Royal Challengers Bangalore' if rcb_wins else 'Mumbai Indians'
winner_conf = int(final_rcb * 100) if rcb_wins else int(final_mi * 100)

# Raipur: par ~175-185, batting friendly with dew, evening match
if rcb_wins:
    rcb_score_mu, mi_score_mu = 188, 168
else:
    rcb_score_mu, mi_score_mu = 168, 188
rcb_score = np.random.normal(loc=rcb_score_mu, scale=20, size=10000)
mi_score = np.random.normal(loc=mi_score_mu, scale=20, size=10000)
if rcb_wins:
    margin = rcb_score - mi_score
else:
    margin = mi_score - rcb_score
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
