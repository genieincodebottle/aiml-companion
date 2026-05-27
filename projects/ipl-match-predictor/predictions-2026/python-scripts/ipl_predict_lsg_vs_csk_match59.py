"""
Lucknow Super Giants vs Chennai Super Kings - Match 59, IPL 2026 - May 15
ML Prediction Script using Ensemble Models
Venue: Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow (LSG home)
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
    # Updated Elo as of May 15 (post matches 57-58)
    # LSG: 6 pts/11 (3W-8L), #10, ELIMINATED, broke 6-match losing streak with DLS win vs RCB
    # CSK: 12 pts (5th-ish), 3-match winning streak (MI, DC, LSG), playoff hopeful
    elo_ratings = {
        'Royal Challengers Bangalore': 1790,
        'Gujarat Titans': 1770,
        'Mumbai Indians': 1480,
        'Punjab Kings': 1735,
        'Rajasthan Royals': 1660,
        'Chennai Super Kings': 1680,           # nudged up on 3-match win streak
        'Sunrisers Hyderabad': 1765,
        'Delhi Capitals': 1655,
        'Kolkata Knight Riders': 1640,
        'Lucknow Super Giants': 1430,          # bottom of table, eliminated
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

    features['home_advantage'] = 1.00      # Ekana is a strong home venue (slow, alien to away batters)
    features['toss_factor'] = 1.05
    # Ekana: chasing favored 14/26 (54%), dew factor possible
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.72,
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.40,
        'Punjab Kings': 0.30,
        'Rajasthan Royals': 0.35,
        'Chennai Super Kings': 0.70,            # 3 wins in a row, just beat LSG
        'Sunrisers Hyderabad': 0.65,
        'Delhi Capitals': 0.60,
        'Kolkata Knight Riders': 0.65,
        'Lucknow Super Giants': 0.30,           # 1 win in last 7, season over
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Lucknow Super Giants'
away_team = 'Chennai Super Kings'
venue = 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium'

print(f"\n{'='*60}")
print(f"MATCH 59: {home_team} vs {away_team}")
print(f"Venue: Ekana Stadium, Lucknow")
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
print("  + LSG home at Ekana - slow, spin-friendly pitch suits Rathi/Shahbaz: +7%")
adjustments += 0.07
print("  + Mohammed Shami leading attack with new ball at home: +3%")
adjustments += 0.03
print("  + Prince Yadav top-3 Purple Cap contender - wickets at home: +3%")
adjustments += 0.03
print("  + Pooran power at the top, Pant fighting for personal pride: +3%")
adjustments += 0.03
print("  + CSK without Dhoni - middle order leadership void on tricky surface: +3%")
adjustments += 0.03
print("  + Home crowd at Ekana + familiar conditions: +3%")
adjustments += 0.03
print("  + CSK historically poor away from Chepauk - travel + slow deck challenge: +3%")
adjustments += 0.03
print("  + 'Must-win' pressure can BACKFIRE on CSK - choke risk: +2%")
adjustments += 0.02
print("  - CSK on 3-match winning streak (MI 8w, DC 8w, LSG 5w): -3%")
adjustments -= 0.03
print("  - CSK just beat LSG 5 days ago - recent psychological edge: -2%")
adjustments -= 0.02
print("  - LSG already ELIMINATED - motivation question on dead rubber: -2%")
adjustments -= 0.02
print("  - Ruturaj Gaikwad in form, Samson opening - CSK top order firing: -2%")
adjustments -= 0.02
print("  - Wanindu Hasaranga unavailable - removes LSG's mystery spin option: -2%")
adjustments -= 0.02
print("  - Chasing favored at Ekana (14/26, 54%) - dew may aid second innings: 0%")

final_lsg = max(0.10, min(0.90, lsg_prob + adjustments))
final_csk = 1.0 - final_lsg
print(f"\nFinal P(LSG wins): {final_lsg:.1%}")
print(f"Final P(CSK wins): {final_csk:.1%}")

np.random.seed(42)
lsg_wins = final_lsg >= 0.5
winner_abbrev = 'LSG' if lsg_wins else 'CSK'
winner_full = 'Lucknow Super Giants' if lsg_wins else 'Chennai Super Kings'
winner_conf = int(final_lsg * 100) if lsg_wins else int(final_csk * 100)

# Ekana: par 170-180, spin-friendly, slow
if lsg_wins:
    lsg_score_mu, csk_score_mu = 178, 162
else:
    lsg_score_mu, csk_score_mu = 165, 180
lsg_score = np.random.normal(loc=lsg_score_mu, scale=18, size=10000)
csk_score = np.random.normal(loc=csk_score_mu, scale=18, size=10000)
if lsg_wins:
    margin = lsg_score - csk_score
else:
    margin = csk_score - lsg_score
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
