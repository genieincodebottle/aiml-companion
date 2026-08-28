"""
Rajasthan Royals vs Sunrisers Hyderabad - Match 36, IPL 2026 - April 25
ML Prediction Script using Ensemble Models
NOTE: Actual venue is Sawai Mansingh Stadium Jaipur (DB has stale Guwahati - flag for fix)
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
df['team1'] = df['team1'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['team2'] = df['team2'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['winner'] = df['winner'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
print(f"Loaded {len(df)} historical matches (2008-2024)")

def engineer_features(home_team, away_team, venue, historical_df):
    features = {}
    elo_ratings = {
        'Royal Challengers Bangalore': 1755,
        'Gujarat Titans': 1685,
        'Mumbai Indians': 1670,
        'Punjab Kings': 1780,
        'Rajasthan Royals': 1715,              # 5W-2L, NRR +1.42
        'Chennai Super Kings': 1600,
        'Sunrisers Hyderabad': 1690,           # 4W-3L, 3-match streak
        'Delhi Capitals': 1640,
        'Kolkata Knight Riders': 1500,
        'Lucknow Super Giants': 1500,
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
    # Sawai Mansingh: balanced, evening dew helps chasing slightly
    features['chase_factor'] = 1.05

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.50,
        'Mumbai Indians': 0.65,
        'Punjab Kings': 0.95,
        'Rajasthan Royals': 0.70,              # bounced back vs LSG with 40-run W
        'Chennai Super Kings': 0.35,
        'Sunrisers Hyderabad': 0.75,           # 3-match streak (RR, CSK, DC)
        'Delhi Capitals': 0.55,
        'Kolkata Knight Riders': 0.15,
        'Lucknow Super Giants': 0.20,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Rajasthan Royals'
away_team = 'Sunrisers Hyderabad'
venue = 'Sawai Mansingh'

print(f"\n{'='*60}")
print(f"MATCH 36: {home_team} vs {away_team}")
print(f"Venue: Sawai Mansingh Stadium, Jaipur, Date: April 25, 2026")
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
print("  + RR first home game in Jaipur this season, fortress motivation: +4%")
adjustments += 0.04
print("  + RR coming off dominant 40-run win vs LSG, Jadeja 43 + Archer 3/20: +3%")
adjustments += 0.03
print("  + RR sit 2nd-3rd with 5W-2L, 10 pts, NRR +1.42 reflects dominance: +3%")
adjustments += 0.03
print("  + Yashasvi Jaiswal + Vaibhav Sooryavanshi top order match Jaipur batting friendly: +2%")
adjustments += 0.02
print("  + Jofra Archer fit and firing, key vs SRH heavy top order: +2%")
adjustments += 0.02
print("  + Ravindra Jadeja all-round contribution (43 last) on spin-friendly later phase: +2%")
adjustments += 0.02
print("  - SRH on 3-match win streak (beat RR, CSK, DC consecutively): -5%")
adjustments -= 0.05
print("  - Pat Cummins returns to take back captaincy, bowling boost + leadership lift: -4%")
adjustments -= 0.04
print("  - Abhishek Sharma 323 runs (avg 53.83, SR 215.33) plus century vs DC: -4%")
adjustments -= 0.04
print("  - Klaasen 320 runs avg 53, three fifties, devastating middle-overs: -3%")
adjustments -= 0.03
print("  - Eshan Malinga 12 wickets leading SRH pace with Sakib + Hinge support: -3%")
adjustments -= 0.03
print("  - SRH posted 242 vs DC last game, batting depth firing on all cylinders: -2%")
adjustments -= 0.02

final_rr = max(0.10, min(0.90, rr_prob + adjustments))
final_srh = 1.0 - final_rr
print(f"\nFinal P(RR wins):   {final_rr:.1%}")
print(f"Final P(SRH wins):  {final_srh:.1%}")

np.random.seed(42)
rr_wins = final_rr >= 0.5
winner_abbrev = 'RR' if rr_wins else 'SRH'
winner_full = 'Rajasthan Royals' if rr_wins else 'Sunrisers Hyderabad'
winner_conf = int(final_rr * 100) if rr_wins else int(final_srh * 100)

if rr_wins:
    rr_score_mu, srh_score_mu = 185, 168
else:
    rr_score_mu, srh_score_mu = 170, 200
rr_score = np.random.normal(loc=rr_score_mu, scale=23, size=10000)
srh_score = np.random.normal(loc=srh_score_mu, scale=28, size=10000)
if rr_wins:
    margin = rr_score - srh_score
else:
    margin = srh_score - rr_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print(f"KEY FACTORS (Favoring {winner_abbrev}):")
print("="*60)
if rr_wins:
    print("1. RR first home match in Jaipur this season, Sawai Mansingh fortress")
    print("2. Coming off dominant 40-run win vs LSG, Jadeja 43 + Archer 3/20")
    print("3. RR 2nd-3rd with 5W-2L, NRR +1.42 dominance, Jaiswal-Sooryavanshi top")
    print("4. Jofra Archer fit and firing, ideal weapon vs SRH heavy top order")
else:
    print("1. SRH on 3-match win streak (beat RR, CSK, DC consecutively)")
    print("2. Pat Cummins returns to take back captaincy, bowling + leadership lift")
    print("3. Abhishek Sharma 323 runs (avg 53.83, SR 215.33), century vs DC")
    print("4. Klaasen 320 runs avg 53 + Eshan Malinga 12 wickets leading attack")

print(f"\n{'='*60}")
risk_team = 'SRH' if rr_wins else 'RR'
print(f"RISK FACTORS ({risk_team} Could Upset):")
print("="*60)
if rr_wins:
    print("1. SRH 3-match win streak including a previous result vs RR this season")
    print("2. Pat Cummins return restores ace pacer + captain leadership combo")
    print("3. Abhishek Sharma 323 runs at SR 215 plus Klaasen 320 runs middle-over carnage")
    print("4. SRH posted 242 vs DC last game, batting depth firing on all cylinders")
else:
    print("1. RR home advantage at Jaipur Sawai Mansingh, fortress motivation")
    print("2. RR 5W-2L, NRR +1.42, dominant 40-run win vs LSG with Jadeja 43, Archer 3/20")
    print("3. Yashasvi Jaiswal + Vaibhav Sooryavanshi match Jaipur batting friendly conditions")
    print("4. Jofra Archer fit and firing, threatens SRH heavy top order")

print(f"\n{'='*60}")
print("PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: {winner_full} ({winner_abbrev})")
print(f"CONFIDENCE: {winner_conf}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
if rr_wins:
    print(f"REASONING: Rajasthan Royals enter Match 36 as narrow favorites in their first home game of the season at the Sawai Mansingh Stadium, Jaipur. They sit 2nd-3rd in the points table with 5 wins from 7, NRR +1.42, having just delivered a dominant 40-run victory over Lucknow Super Giants in which Ravindra Jadeja contributed a crucial 43 and Jofra Archer ripped through the chase with 3 for 20. The Jaiswal-Sooryavanshi top order suits the balanced Jaipur surface where the average first innings score is 165-175 and stroke play is rewarded once the new ball is gone. Sunrisers Hyderabad arrive on a 3-match win streak having beaten DC, CSK and RR consecutively, with Abhishek Sharma in stunning form (323 runs, SR 215) and Pat Cummins returning to take back captaincy after his back injury. But RR home advantage, Archer fitness, and Jadeja all-round impact tilt the scales narrowly to the hosts.")
else:
    print(f"REASONING: Sunrisers Hyderabad arrive at Sawai Mansingh as narrow favorites despite playing away. They are on a 3-match winning streak having beaten Rajasthan Royals, Chennai Super Kings and Delhi Capitals consecutively, including a 47-run thumping of DC powered by 242 on the board. Abhishek Sharma has scored 323 runs in seven innings at an average of 53.83 and a strike rate of 215.33, including a brilliant 135 off 68 vs DC, while Heinrich Klaasen has 320 runs at an average of 53 and a strike rate of 153, including three fifties. Pat Cummins is cleared to play and resumes captaincy from Ishan Kishan, restoring both an ace pacer and the strategic leadership that carried SRH to last season's heights. Eshan Malinga leads the IPL pace charts with 12 wickets, supported by Sakib Hussain and Praful Hinge. RR counter with home advantage at Jaipur, a recent 40-run win over LSG, Jaiswal-Sooryavanshi at the top, and Archer firing 3 for 20, but the combined weight of SRH's batting carnage, momentum, and Cummins' return tilts the scales narrowly to the visitors.")
print(f"{'='*60}\n")
print("DB UPDATE VALUES:")
print(f"  ml_winner: '{winner_abbrev}'")
print(f"  ml_confidence: {winner_conf}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
