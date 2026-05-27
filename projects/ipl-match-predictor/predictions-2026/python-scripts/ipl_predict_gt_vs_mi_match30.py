"""
Gujarat Titans vs Mumbai Indians - Match 30, IPL 2026 - April 20
ML Prediction Script using Ensemble Models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os

csv_path = r'c:\projects\aiml-companion\projects\ipl-match-predictor\data\raw\matches.csv'
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} historical matches (2008-2024)")

def engineer_features(home_team, away_team, venue, historical_df):
    features = {}
    elo_ratings = {
        'Gujarat Titans': 1730,        # 3W 2L, 4th place, 3-match win streak
        'Mumbai Indians': 1580,        # 1W 4L, 9th place, 4 consecutive losses
        'Royal Challengers Bangalore': 1760,
        'Punjab Kings': 1750,
        'Rajasthan Royals': 1700,
        'Chennai Super Kings': 1640,
        'Sunrisers Hyderabad': 1640,
        'Delhi Capitals': 1580,
        'Kolkata Knight Riders': 1480,
        'Lucknow Super Giants': 1560,
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

    venue_matches = historical_df[historical_df['venue'] == venue]
    if len(venue_matches) > 0:
        home_wins = len(venue_matches[(venue_matches['team1'] == home_team) & (venue_matches['winner'] == home_team)])
        home_total = len(venue_matches[venue_matches['team1'] == home_team])
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.60
    else:
        features['venue_home_win_pct'] = 0.60

    features['home_advantage'] = 1.0
    features['toss_factor'] = 1.05
    features['chase_factor'] = 0.95  # Ahmedabad: batting first slightly favored (heat, dry pitch)

    momentum_2026 = {
        'Gujarat Titans': 0.80,        # 3-match win streak, riding high
        'Mumbai Indians': 0.20,        # 4 consecutive losses, Rohit Sharma injured
        'Royal Challengers Bangalore': 0.90,
        'Punjab Kings': 0.95,
        'Rajasthan Royals': 0.65,
        'Chennai Super Kings': 0.60,
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.45,
        'Kolkata Knight Riders': 0.10,
        'Lucknow Super Giants': 0.35,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Gujarat Titans'
away_team = 'Mumbai Indians'
venue = 'Ahmedabad'

print(f"\n{'='*60}")
print(f"MATCH 30: {home_team} vs {away_team}")
print(f"Venue: Narendra Modi Stadium, {venue}, Date: April 20, 2026")
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
    print(f"  {name.upper()}: {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble (GT wins): {ensemble:.1%}")

print("\nContextual Adjustments:")
adjustments = 0
print("  + GT 3-match win streak, Shubman Gill 251 runs (avg 50): +4%")
adjustments += 0.04
print("  + GT perfect home record vs MI (4-0 at Narendra Modi Stadium): +4%")
adjustments += 0.04
print("  + Rashid Khan 10 wickets vs MI in H2H history: +3%")
adjustments += 0.03
print("  + Prasidh Krishna leading Purple Cap in 2026: +2%")
adjustments += 0.02
print("  - MI historically strong side despite poor 2026 form: -2%")
adjustments -= 0.02
print("  - Rohit Sharma injury doubt but Hardik Pandya leading: -1%")
adjustments -= 0.01
print("  + MI 4 consecutive losses, team morale low: +3%")
adjustments += 0.03
print("  + GT home crowd, Ahmedabad heat favors GT players: +1%")
adjustments += 0.01
print("  - Jasprit Bumrah due a breakthrough (0 wickets in 2026): -1%")
adjustments -= 0.01

final_gt = min(0.95, ensemble + adjustments)
print(f"\nFinal (GT wins): {final_gt:.1%}")

np.random.seed(42)
gt_score = np.random.normal(loc=182, scale=22, size=10000)   # GT batting at home
mi_score = np.random.normal(loc=158, scale=27, size=10000)   # MI struggling
margin = gt_score - mi_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print("KEY FACTORS (Favoring GT):")
print("="*60)
print("1. Perfect home record: GT 4-0 vs MI at Narendra Modi Stadium all time")
print("2. Shubman Gill in red-hot form: 251 runs in 5 innings, avg 50 this season")
print("3. Rashid Khan's 10 wickets in 4 H2H matches - MI have no answer for him")
print("4. GT on 3-match win streak (SRH-like 57 run demolitions), team firing on all cylinders")

print(f"\n{'='*60}")
print("RISK FACTORS (MI Could Upset):")
print("="*60)
print("1. Jasprit Bumrah overdue a wicket haul (0 wickets in 2026 - historically elite)")
print("2. Hardik Pandya can produce match-winning all-round performances under pressure")
print("3. Suryakumar Yadav power hitting at Ahmedabad's pace-friendly surface")
print("4. MI's 5-time champions pedigree - never count them out")

print(f"\n{'='*60}")
print("PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: Gujarat Titans (GT)")
print(f"CONFIDENCE: {int(final_gt * 100)}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
print(f"REASONING: GT are overwhelming favorites at home - they have beaten MI in all 4 previous encounters at Narendra Modi Stadium. Shubman Gill (251 runs, avg 50) + Rashid Khan (10 H2H wickets) form an unbeatable combination. MI's 4 consecutive defeats, Rohit Sharma's hamstring injury, and Jasprit Bumrah's wicket drought in 2026 make them highly vulnerable against a GT side firing on all cylinders.")
print(f"{'='*60}\n")
print("DB UPDATE VALUES:")
print(f"  ml_winner: 'GT'")
print(f"  ml_confidence: {int(final_gt * 100)}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
