"""
Delhi Capitals vs Punjab Kings - Match 35, IPL 2026 - April 25
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
df['team1'] = df['team1'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['team2'] = df['team2'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['winner'] = df['winner'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
print(f"Loaded {len(df)} historical matches (2008-2024)")

def engineer_features(home_team, away_team, venue, historical_df):
    features = {}
    # Elo ratings as of April 25, Match 35 (after Match 34 results factored)
    elo_ratings = {
        'Royal Challengers Bangalore': 1755,
        'Gujarat Titans': 1685,
        'Mumbai Indians': 1670,
        'Punjab Kings': 1780,                  # Top, unbeaten 5W-1NR, NRR +1.42
        'Rajasthan Royals': 1715,              # 5W-2L, beat LSG by 40
        'Chennai Super Kings': 1600,
        'Sunrisers Hyderabad': 1690,           # 4W-3L, 3-match streak
        'Delhi Capitals': 1640,                # 15 pts, 5W-3L mid-table
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
    # Arun Jaitley: balanced, slight edge to chasing late, spin grips
    features['chase_factor'] = 1.02

    # Momentum reflects current IPL 2026 form (as of April 25, 2026)
    momentum_2026 = {
        'Royal Challengers Bangalore': 0.65,
        'Gujarat Titans': 0.50,
        'Mumbai Indians': 0.65,
        'Punjab Kings': 0.95,                  # unbeaten
        'Rajasthan Royals': 0.70,              # bounced back vs LSG
        'Chennai Super Kings': 0.35,
        'Sunrisers Hyderabad': 0.75,           # 3-match streak
        'Delhi Capitals': 0.55,                # mixed: beat RCB, lost SRH
        'Kolkata Knight Riders': 0.15,
        'Lucknow Super Giants': 0.20,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Delhi Capitals'
away_team = 'Punjab Kings'
venue = 'Arun Jaitley'

print(f"\n{'='*60}")
print(f"MATCH 35: {home_team} vs {away_team}")
print(f"Venue: Arun Jaitley Stadium, Delhi, Date: April 25, 2026")
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
    print(f"  {name.upper()} P(DC wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(DC wins): {ensemble:.1%}")
dc_prob = ensemble

print("\nContextual Adjustments (positive = favor DC):")
adjustments = 0
print("  + DC at home Arun Jaitley with familiar conditions, KL Rahul wicketkeeper anchor: +3%")
adjustments += 0.03
print("  + DC beat RCB by 6 wkts on Apr 18, Axar Patel MVP, captaincy gelling: +3%")
adjustments += 0.03
print("  + Kuldeep Yadav + Axar spin twins on slow-turning Delhi pitch (late grip): +3%")
adjustments += 0.03
print("  + DC won toss, bat first to set total in 35-42C heat draining chasers: +2%")
adjustments += 0.02
print("  + Stubbs/Miller power finishers on short-boundary Delhi outfield: +2%")
adjustments += 0.02
print("  - PBKS unbeaten table-toppers (5W-1NR), NRR +1.42, only loss-free side: -8%")
adjustments -= 0.08
print("  - Shreyas Iyer in elite captain form: 203 runs in 5, prime catch leadership: -4%")
adjustments -= 0.04
print("  - Priyansh Arya 93 off 37 + Cooper Connolly 87 off 46 most recently vs LSG: -4%")
adjustments -= 0.04
print("  - Arshdeep Singh + Marco Jansen left-arm pace combo destabilises RH-heavy DC top: -3%")
adjustments -= 0.03
print("  - Yuzvendra Chahal back at his hunting ground (Delhi), spin matching pitch: -3%")
adjustments -= 0.03
print("  - DC lost last to SRH by 47 runs, top-order frailty exposed: -2%")
adjustments -= 0.02

final_dc = max(0.10, min(0.90, dc_prob + adjustments))
final_pbks = 1.0 - final_dc
print(f"\nFinal P(DC wins):    {final_dc:.1%}")
print(f"Final P(PBKS wins):  {final_pbks:.1%}")

np.random.seed(42)
dc_wins = final_dc >= 0.5
winner_abbrev = 'DC' if dc_wins else 'PBKS'
winner_full = 'Delhi Capitals' if dc_wins else 'Punjab Kings'
winner_conf = int(final_dc * 100) if dc_wins else int(final_pbks * 100)

if dc_wins:
    dc_score_mu, pbks_score_mu = 195, 175
else:
    dc_score_mu, pbks_score_mu = 175, 200
dc_score = np.random.normal(loc=dc_score_mu, scale=24, size=10000)
pbks_score = np.random.normal(loc=pbks_score_mu, scale=26, size=10000)
if dc_wins:
    margin = dc_score - pbks_score
else:
    margin = pbks_score - dc_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print(f"KEY FACTORS (Favoring {winner_abbrev}):")
print("="*60)
if dc_wins:
    print("1. DC at home Arun Jaitley, KL Rahul wicketkeeper anchor + familiar conditions")
    print("2. Beat RCB by 6 wkts in last home game, Axar Patel MVP captaincy clicking")
    print("3. Kuldeep Yadav + Axar Patel spin pair on slow Delhi deck that grips late")
    print("4. Stubbs/Miller power finishers on short-boundary Delhi outfield")
else:
    print("1. PBKS unbeaten table-toppers (5W-1NR), NRR +1.42, only loss-free side in IPL 2026")
    print("2. Shreyas Iyer in elite captain form: 203 runs in 5 innings, prime catch leadership")
    print("3. Priyansh Arya (93 off 37) + Cooper Connolly (87 off 46) explosive top order")
    print("4. Arshdeep + Jansen left-arm pace combo + Chahal spin on home ground")

print(f"\n{'='*60}")
risk_team = 'PBKS' if dc_wins else 'DC'
print(f"RISK FACTORS ({risk_team} Could Upset):")
print("="*60)
if dc_wins:
    print("1. PBKS unbeaten run (5W-1NR) and NRR +1.42, only undefeated side")
    print("2. Iyer 203 runs + Priyansh Arya/Connolly 180-run stand vs LSG firepower")
    print("3. Arshdeep-Jansen new-ball + Chahal spin on his Delhi home turf")
    print("4. PBKS momentum and depth give them edge in pressure chases")
else:
    print("1. DC home advantage at Arun Jaitley, KL Rahul anchor role")
    print("2. Beat RCB by 6 wkts last home, Axar captaincy MVP performance")
    print("3. Kuldeep + Axar spin twins on slow Delhi pitch ideal vs PBKS top order")
    print("4. Delhi 35-42C heat factor draining away side in afternoon match")

print(f"\n{'='*60}")
print("PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: {winner_full} ({winner_abbrev})")
print(f"CONFIDENCE: {winner_conf}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
if dc_wins:
    print(f"REASONING: Delhi Capitals enter Match 35 as narrow favorites at the Arun Jaitley Stadium where home conditions and a settled spin attack tilt the balance. KL Rahul anchors a top order that beat RCB by 6 wickets just a week ago, with Axar Patel being named Cricinfo MVP for his captaincy and all-round contribution. The slow-turning Delhi surface, which grips noticeably as the game wears on, perfectly suits the Kuldeep Yadav and Axar Patel spin pair. Tristan Stubbs and David Miller provide power finishing on a short-boundary outfield, while DC won the toss and chose to bat first to set a chasing target in punishing 35-42C heat. Punjab Kings are dangerous and unbeaten with five wins and one no-result, sitting top of the table with NRR +1.42, but a flat away surface where Chahal will hunt and the heat factor on visitors keep DC slightly ahead.")
else:
    print(f"REASONING: Punjab Kings arrive at the Arun Jaitley Stadium as the only unbeaten side in IPL 2026, holding five wins, one no-result and a commanding NRR of +1.42 at the top of the table. Captain Shreyas Iyer has scored 203 runs in five innings and his leadership has been hailed by coaching staff as the catalyst for this turnaround. Priyansh Arya (93 off 37) and Cooper Connolly (87 off 46) just stitched a 182-run stand to dismantle LSG bowling and post 254/7, exposing the depth and intent at the top. The left-arm pace pairing of Arshdeep Singh and Marco Jansen unsettles right-hand-heavy DC top orders, while Yuzvendra Chahal returns to his hunting ground in Delhi where the slow surface grips for spin. DC counter with home advantage, the Axar-Kuldeep spin twins, KL Rahul anchoring, and a recent six-wicket win over RCB, but the combined weight of unbeaten momentum, captain in form, top-order firepower and bowling balance favors PBKS.")
print(f"{'='*60}\n")
print("DB UPDATE VALUES:")
print(f"  ml_winner: '{winner_abbrev}'")
print(f"  ml_confidence: {winner_conf}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
