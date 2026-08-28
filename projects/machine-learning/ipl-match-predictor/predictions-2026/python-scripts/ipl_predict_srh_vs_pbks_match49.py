"""
Sunrisers Hyderabad vs Punjab Kings - Match 49, IPL 2026 - May 5/6
ML Prediction Script using Ensemble Models
Venue: Rajiv Gandhi International Stadium, Hyderabad
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
    # Updated Elo as of May 5 (after 47 matches)
    # SRH: 12 pts, 3rd place, 5 consecutive wins (HOT FORM)
    # PBKS: 13 pts, 1st place, 6W-2L-1NR, NRR +0.855, lost 2 straight
    elo_ratings = {
        'Royal Challengers Bangalore': 1745,
        'Gujarat Titans': 1735,
        'Mumbai Indians': 1490,
        'Punjab Kings': 1775,               # league leader but lost 2 straight
        'Rajasthan Royals': 1700,
        'Chennai Super Kings': 1645,
        'Sunrisers Hyderabad': 1760,        # 5-match win streak, ~242 vs DC
        'Delhi Capitals': 1690,
        'Kolkata Knight Riders': 1515,
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

    venue_matches = historical_df[historical_df['venue'].str.contains(venue, na=False, case=False)]
    if len(venue_matches) > 0:
        home_wins = len(venue_matches[(venue_matches['team1'] == home_team) & (venue_matches['winner'] == home_team)])
        home_total = len(venue_matches[venue_matches['team1'] == home_team])
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.55
    else:
        features['venue_home_win_pct'] = 0.55

    features['home_advantage'] = 1.0
    features['toss_factor'] = 1.05
    # Uppal: batting paradise, last 10 T20s avg 194 first innings, 14/14 favored batters
    features['chase_factor'] = 1.10

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.75,
        'Mumbai Indians': 0.20,
        'Punjab Kings': 0.55,               # league leader but lost 2 straight
        'Rajasthan Royals': 0.40,
        'Chennai Super Kings': 0.45,
        'Sunrisers Hyderabad': 0.85,        # 5-match win streak
        'Delhi Capitals': 0.55,
        'Kolkata Knight Riders': 0.45,
        'Lucknow Super Giants': 0.15,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Sunrisers Hyderabad'
away_team = 'Punjab Kings'
venue = 'Rajiv Gandhi International Stadium'

print(f"\n{'='*60}")
print(f"MATCH 49: {home_team} vs {away_team}")
print(f"Venue: Rajiv Gandhi International Stadium, Hyderabad")
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
    print(f"  {name.upper()} P(SRH wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(SRH wins): {ensemble:.1%}")
srh_prob = ensemble

print("\nContextual Adjustments (positive = favor SRH):")
adjustments = 0
print("  + Home advantage at Uppal, batting paradise SRH knows best: +4%")
adjustments += 0.04
print("  + SRH on 5-match win streak (HOT FORM), highest momentum in league: +5%")
adjustments += 0.05
print("  + SRH H2H dominance: 17-8 all-time (68% win rate): +4%")
adjustments += 0.04
print("  + Abhishek Sharma (440 runs, 135* vs DC) explosive top-order: +4%")
adjustments += 0.04
print("  + Heinrich Klaasen (414 runs) middle-order anchor + finisher: +3%")
adjustments += 0.03
print("  + Travis Head powerplay aggression on flat Uppal track: +3%")
adjustments += 0.03
print("  + PBKS lost 2 straight (vs GT May 3) - momentum slump: +2%")
adjustments += 0.02
print("  - PBKS league-table leader (13 pts) - resilient under pressure: -3%")
adjustments -= 0.03
print("  - Eshan Malinga purple cap leader (18 wkts, eco 9.09) wickets matter: -2%")
adjustments -= 0.02
print("  - Arshdeep Singh + Kagiso Rabada lethal new ball pair vs SRH top: -3%")
adjustments -= 0.03
print("  - Yuzvendra Chahal threat in middle overs, exploited SRH Apr 11: -2%")
adjustments -= 0.02
print("  - Shreyas Iyer captaincy + form, PBKS batting depth (Stoinis, Shashank): -3%")
adjustments -= 0.03

final_srh = max(0.10, min(0.90, srh_prob + adjustments))
final_pbks = 1.0 - final_srh
print(f"\nFinal P(SRH wins):  {final_srh:.1%}")
print(f"Final P(PBKS wins): {final_pbks:.1%}")

np.random.seed(42)
srh_wins = final_srh >= 0.5
winner_abbrev = 'SRH' if srh_wins else 'PBKS'
winner_full = 'Sunrisers Hyderabad' if srh_wins else 'Punjab Kings'
winner_conf = int(final_srh * 100) if srh_wins else int(final_pbks * 100)

if srh_wins:
    srh_score_mu, pbks_score_mu = 198, 178
else:
    srh_score_mu, pbks_score_mu = 175, 195
srh_score = np.random.normal(loc=srh_score_mu, scale=22, size=10000)
pbks_score = np.random.normal(loc=pbks_score_mu, scale=22, size=10000)
if srh_wins:
    margin = srh_score - pbks_score
else:
    margin = pbks_score - srh_score
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
