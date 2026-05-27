"""
Kolkata Knight Riders vs Mumbai Indians - Match 65, IPL 2026 - May 20 (7:30 PM IST)
ML Prediction Script using Ensemble Models
Venue: Eden Gardens, Kolkata (KKR home)
Context:
  KKR 12P 5W-6L-1NR = 11 pts, 8th place, MUST WIN to keep slim playoff hope alive.
    Hot form: 5 wins in last 6.
    Captain: Ajinkya Rahane (recent form patchy but contributed 67(40) in M2).
    Top scorer: Angkrish Raghuvanshi 422 runs avg 42.2.
    Finn Allen destructive: 93(35) in last match vs GT, SR 233 across 240 runs in season.
    Bowling: Sunil Narine, Varun Chakravarthy, Kartik Tyagi.
  MI 12P 4W-8L = 8 pts, ELIMINATED, playing for pride/spoiler role.
    Top order back in form (Rohit, Rickelton, Tilak, SKY returning).
    Hardik Pandya (c) and Suryakumar Yadav set to return.
    Bowling: Jasprit Bumrah, Trent Boult, Shardul Thakur (took 3/39 in M2).
  H2H all-time: MI 25-11 KKR (MI dominant ~69%).
  H2H recent: KKR won both 2024 fixtures. MI won M2 of this season at Wankhede by 6 wkts (chased 221).
  Eden Gardens IPL 2026: avg 1st innings 202, par 205-215, chasing edge ~57%, heavy dew 12th-14th over.
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
    # Updated Elo as of May 19 evening (post-M64)
    elo_ratings = {
        'Royal Challengers Bangalore': 1830,
        'Gujarat Titans': 1815,
        'Sunrisers Hyderabad': 1745,
        'Chennai Super Kings': 1710,
        'Punjab Kings': 1685,
        'Delhi Capitals': 1665,
        'Kolkata Knight Riders': 1660,   # bumped after 5W in 6 surge
        'Rajasthan Royals': 1640,        # won M64 vs LSG
        'Mumbai Indians': 1500,           # 4W from 12, eliminated
        'Lucknow Super Giants': 1410,    # bottom of table
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

    features['home_advantage'] = 1.08   # Eden Gardens true home for KKR, vocal crowd
    features['toss_factor'] = 1.06
    features['chase_factor'] = 1.07     # heavy dew makes chasing easier at Eden Gardens

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.85,
        'Gujarat Titans': 0.78,
        'Sunrisers Hyderabad': 0.70,
        'Chennai Super Kings': 0.65,
        'Punjab Kings': 0.22,
        'Mumbai Indians': 0.52,           # win vs PBKS, top order back, SKY+Hardik return
        'Delhi Capitals': 0.55,
        'Rajasthan Royals': 0.30,
        'Kolkata Knight Riders': 0.80,    # 5W in last 6 surge, Allen hot
        'Lucknow Super Giants': 0.22,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Kolkata Knight Riders'
away_team = 'Mumbai Indians'
venue = 'Eden Gardens'

print(f"\n{'='*60}")
print(f"MATCH 65: {home_team} vs {away_team}")
print(f"Venue: Eden Gardens, Kolkata")
print(f"Time: 7:30 PM IST (evening, heavy dew expected)")
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
    print(f"  {name.upper()} P(KKR wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(KKR wins): {ensemble:.1%}")
kkr_prob = ensemble

print("\nContextual Adjustments (positive = favor KKR):")
adjustments = 0
print("  + KKR at home Eden Gardens with vocal sellout crowd, win-or-out mode: +5%")
adjustments += 0.05
print("  + KKR red-hot 5 wins in last 6, momentum strongly with them: +5%")
adjustments += 0.05
print("  + Playoff stakes - KKR fighting for survival vs MI eliminated/spoiler: +4%")
adjustments += 0.04
print("  + Finn Allen destructive form (93 off 35 last game, SR 233 season): +3%")
adjustments += 0.03
print("  + Eden Gardens flat 205-215 par track favors KKR power hitters: +2%")
adjustments += 0.02
print("  + KKR won both 2024 H2H vs MI - mental edge over recent samples: +2%")
adjustments += 0.02
print("  - MI top order Rohit+Rickelton back in form, just smashed PBKS: -4%")
adjustments -= 0.04
print("  - Suryakumar Yadav and Hardik Pandya returning - huge MI batting boost: -4%")
adjustments -= 0.04
print("  - Jasprit Bumrah + Trent Boult elite pace attack on dew-soaked deck: -3%")
adjustments -= 0.03
print("  - MI dominate KKR all-time 25-11, M2 this season MI chased 221: -3%")
adjustments -= 0.03
print("  - Ajinkya Rahane (KKR captain) off-colour after early games: -2%")
adjustments -= 0.02
print("  - Heavy dew from 12-14th over wipes out Narine/Varun spin threat: -2%")
adjustments -= 0.02

final_kkr = max(0.10, min(0.90, kkr_prob + adjustments))
final_mi = 1.0 - final_kkr
print(f"\nFinal P(KKR wins): {final_kkr:.1%}")
print(f"Final P(MI wins): {final_mi:.1%}")

np.random.seed(42)
kkr_wins = final_kkr >= 0.5
winner_abbrev = 'KKR' if kkr_wins else 'MI'
winner_full = 'Kolkata Knight Riders' if kkr_wins else 'Mumbai Indians'
winner_conf = int(final_kkr * 100) if kkr_wins else int(final_mi * 100)

# Eden Gardens IPL 2026: par 205-215, chase-friendly with dew
if kkr_wins:
    kkr_score_mu, mi_score_mu = 208, 188
else:
    kkr_score_mu, mi_score_mu = 192, 215
kkr_score = np.random.normal(loc=kkr_score_mu, scale=18, size=10000)
mi_score = np.random.normal(loc=mi_score_mu, scale=18, size=10000)
if kkr_wins:
    margin = kkr_score - mi_score
else:
    margin = mi_score - kkr_score
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
