"""
Kolkata Knight Riders vs Gujarat Titans - Match 60, IPL 2026 - May 16
ML Prediction Script using Ensemble Models
Venue: Eden Gardens, Kolkata (KKR home)
Context: KKR must-win (playoffs hanging by thread, ~4W) vs GT (16 pts, already qualified, on 5-match winning streak)
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
    # Updated Elo as of May 15 (after matches 57-59)
    # GT: 16 pts/12 (8W-4L), #1, already qualified, 5-match winning streak
    # KKR: ~4W from 12 (~8L), #7/8, playoff hopes slim, just lost to RCB after 4W streak
    elo_ratings = {
        'Royal Challengers Bangalore': 1810,         # top after KKR win
        'Gujarat Titans': 1820,                       # top table, 5-streak
        'Mumbai Indians': 1480,
        'Punjab Kings': 1720,
        'Rajasthan Royals': 1650,
        'Chennai Super Kings': 1690,
        'Sunrisers Hyderabad': 1730,                  # took beating from GT
        'Delhi Capitals': 1650,
        'Kolkata Knight Riders': 1635,                # broke 4W streak with loss
        'Lucknow Super Giants': 1420,
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

    features['home_advantage'] = 1.05       # Eden Gardens is a strong home venue
    features['toss_factor'] = 1.05
    # Eden Gardens: chasing favored 57/102 (~56%), dew factor high in May evening
    features['chase_factor'] = 1.10

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.78,
        'Gujarat Titans': 0.85,                       # 5-match win streak, top of table
        'Mumbai Indians': 0.40,
        'Punjab Kings': 0.30,
        'Rajasthan Royals': 0.30,
        'Chennai Super Kings': 0.72,
        'Sunrisers Hyderabad': 0.50,
        'Delhi Capitals': 0.55,
        'Kolkata Knight Riders': 0.55,                # 4W, 1L recently - decent form but inconsistent overall
        'Lucknow Super Giants': 0.28,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Kolkata Knight Riders'
away_team = 'Gujarat Titans'
venue = 'Eden Gardens'

print(f"\n{'='*60}")
print(f"MATCH 60: {home_team} vs {away_team}")
print(f"Venue: Eden Gardens, Kolkata")
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
print("  + KKR home at Eden Gardens - massive crowd, familiar conditions: +6%")
adjustments += 0.06
print("  + KKR must-win desperation - playoffs on the line: +4%")
adjustments += 0.04
print("  + Eden Gardens batting friendly, suits Russell/Rinku finishers: +3%")
adjustments += 0.03
print("  + Dew factor at Eden in May - chase support helps if KKR bowls first: +2%")
adjustments += 0.02
print("  + Sunil Narine spin at home venue + new ball threat: +3%")
adjustments += 0.03
print("  + Andre Russell-Rinku Singh power finish, KKR coming off 4W streak before RCB: +3%")
adjustments += 0.03
print("  + GT may rotate squad - already qualified, rest key players for playoffs: +3%")
adjustments += 0.03
print("  - GT on 5-match winning streak (RCB, PBKS, RR, SRH, +1), peak form: -4%")
adjustments -= 0.04
print("  - GT top of table (16 pts), strongest team in IPL 2026: -3%")
adjustments -= 0.03
print("  - GT bowling unit best in IPL 2026 - Rashid + pace battery dangerous: -3%")
adjustments -= 0.03
print("  - GT historical edge at Eden Gardens (2-0), H2H lead 4-1 overall: -2%")
adjustments -= 0.02
print("  - Sai Sudharsan (759 runs) + Gill (717 runs) - top of orange cap race: -3%")
adjustments -= 0.03
print("  - KKR just lost to RCB (Kohli 105*), batting collapse risk under pressure: -2%")
adjustments -= 0.02
print("  - Varun Chakaravarthy fitness doubt - KKR's mystery spinner weakened: -2%")
adjustments -= 0.02
print("  - KKR season inconsistency - 4W streak followed by loss shows fragility: -1%")
adjustments -= 0.01

final_kkr = max(0.10, min(0.90, kkr_prob + adjustments))
final_gt = 1.0 - final_kkr
print(f"\nFinal P(KKR wins): {final_kkr:.1%}")
print(f"Final P(GT wins): {final_gt:.1%}")

np.random.seed(42)
kkr_wins = final_kkr >= 0.5
winner_abbrev = 'KKR' if kkr_wins else 'GT'
winner_full = 'Kolkata Knight Riders' if kkr_wins else 'Gujarat Titans'
winner_conf = int(final_kkr * 100) if kkr_wins else int(final_gt * 100)

# Eden Gardens: par 180-195, batting friendly, dew aids chasers
if kkr_wins:
    kkr_score_mu, gt_score_mu = 195, 178
else:
    kkr_score_mu, gt_score_mu = 175, 195
kkr_score = np.random.normal(loc=kkr_score_mu, scale=18, size=10000)
gt_score = np.random.normal(loc=gt_score_mu, scale=18, size=10000)
if kkr_wins:
    margin = kkr_score - gt_score
else:
    margin = gt_score - kkr_score
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
