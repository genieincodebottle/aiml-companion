"""
Sunrisers Hyderabad vs Royal Challengers Bengaluru - Match 67, IPL 2026 - May 22 (7:30 PM IST)
ML Prediction Script using Ensemble Models
Venue: Rajiv Gandhi International Stadium, Hyderabad (SRH home)

Context (web-researched 2026-05-22):
  RCB 14P 9W-5L = 18 pts, #1 in points table, defending champions, ALREADY QUALIFIED.
    Hot form: 3-match winning streak going into this clash.
    Captain: Rajat Patidar. Top: Virat Kohli (ORANGE CAP - 484 runs in 12 @ 53.78, SR 165.76, 1*100, 3*50).
    Opener: Jacob Bethell + Devdutt Padikkal. Finisher: Tim David.
    WK-bat: Jitesh Sharma. Spin: Krunal Pandya. Pace: Josh Hazlewood + Bhuvneshwar Kumar + Rasikh Salam.
    Playing for top-spot lock + Qualifier 1 home-leg advantage.

  SRH 13P 8W-4L (1NR) = 16 pts, currently #2-3, ALREADY QUALIFIED for playoffs.
    Form: 5-match winning streak halted by KKR (7-wkt loss in reverse fixture).
    Captain: Pat Cummins. Openers: Abhishek Sharma + Travis Head (323 runs in 10 @ 32.30, recent 76(30) vs MI).
    Power middle: Heinrich Klaasen, Ishan Kishan (wk), Nitish Reddy.
    Pace: Cummins (c) + Eshan Malinga + Sakib Hussain.
    Needs win to claim top-2 seeding (Q1 vs Eliminator difference).

  H2H all-time: SRH 14 - RCB 12 (27 matches, 1 NR). SRH historically a bogey team for RCB.

  Rajiv Gandhi International Stadium IPL 2026:
    - Avg 1st innings ~202 (one of highest in IPL 2026)
    - Batting-friendly flat surface, true bounce, good carry
    - Bat-first slightly favored: 3 of 4 wins to team batting first; 3 of 5 chases lost
    - Dew minimal in May Hyderabad (peak summer) - bat first benefit holds even in evening
    - SRH's blueprint = ultra-aggressive bat-first 220+ at home, defended by Cummins-led pace
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
    # Updated Elo as of May 22 morning (post-M66 GT vs CSK)
    elo_ratings = {
        'Royal Challengers Bangalore': 1840,   # +10 #1 table 18 pts, 3-match streak, defending champs
        'Gujarat Titans': 1825,
        'Sunrisers Hyderabad': 1755,           # 16 pts but recent KKR loss tempers Elo
        'Chennai Super Kings': 1695,
        'Punjab Kings': 1685,
        'Delhi Capitals': 1665,
        'Kolkata Knight Riders': 1675,         # +20 beat SRH M65 (reverse) by 7 wkts
        'Rajasthan Royals': 1640,
        'Mumbai Indians': 1505,
        'Lucknow Super Giants': 1410,
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

    features['home_advantage'] = 1.08         # Rajiv Gandhi - true SRH fortress, packed crowd
    features['toss_factor'] = 1.05
    features['chase_factor'] = 0.96           # bat-first slight edge per 2026 trend (dew low in May)

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.86,   # 3-match streak, Kohli Orange Cap form
        'Gujarat Titans': 0.85,
        'Sunrisers Hyderabad': 0.75,           # 5-match streak interrupted by KKR but core hot
        'Chennai Super Kings': 0.38,
        'Punjab Kings': 0.22,
        'Mumbai Indians': 0.52,
        'Delhi Capitals': 0.55,
        'Rajasthan Royals': 0.30,
        'Kolkata Knight Riders': 0.72,
        'Lucknow Super Giants': 0.22,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Sunrisers Hyderabad'
away_team = 'Royal Challengers Bangalore'
venue = 'Rajiv Gandhi International Stadium'

print(f"\n{'='*60}")
print(f"MATCH 67: {home_team} vs {away_team}")
print(f"Venue: Rajiv Gandhi International Stadium, Hyderabad")
print(f"Time: 7:30 PM IST (evening, dew minimal - May heat)")
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
print("  + SRH home Rajiv Gandhi avg 1st inn 202 - their ultra-aggressive bat-first blueprint fits venue: +3%")
adjustments += 0.03
print("  + SRH 5-match winning streak (only ended by KKR last out) - core batting unit in groove: +2%")
adjustments += 0.02
print("  + H2H all-time SRH 14-12 RCB - slight bogey edge for SRH historically: +2%")
adjustments += 0.02
print("  + Klaasen + Head + Abhishek Sharma trio devastating on flat Hyderabad track: +2%")
adjustments += 0.02
print("  + Cummins (c) home spell with new ball + Malinga at death = wicket-taking blueprint: +1%")
adjustments += 0.01
print("  - RCB #1 in table 18 pts (4 wins ahead of SRH in same span), defending champs: -8%")
adjustments -= 0.08
print("  - Virat Kohli ORANGE CAP form (484 runs @ 53.78, SR 165.76, 100* vs KKR) anchors RCB chase: -5%")
adjustments -= 0.05
print("  - Hazlewood + Bhuvi opening burst neutralises SRH explosive PP - top-tier attack: -4%")
adjustments -= 0.04
print("  - SRH coming off KKR loss - momentum break + Cummins-led pace under pressure: -3%")
adjustments -= 0.03
print("  - Krunal Pandya middle-overs control + Patidar tactical sharpness in big games: -2%")
adjustments -= 0.02
print("  - RCB just on 3-match winning streak going in, peaking at the right time: -2%")
adjustments -= 0.02

final_srh = max(0.10, min(0.90, srh_prob + adjustments))
final_rcb = 1.0 - final_srh
print(f"\nFinal P(SRH wins): {final_srh:.1%}")
print(f"Final P(RCB wins): {final_rcb:.1%}")

np.random.seed(42)
srh_wins = final_srh >= 0.5
winner_abbrev = 'SRH' if srh_wins else 'RCB'
winner_full = 'Sunrisers Hyderabad' if srh_wins else 'Royal Challengers Bangalore'
winner_conf = int(final_srh * 100) if srh_wins else int(final_rcb * 100)

# Rajiv Gandhi IPL 2026: 1st inn avg ~202, par 200-215, bat-first slight edge
if srh_wins:
    srh_score_mu, rcb_score_mu = 210, 192
else:
    srh_score_mu, rcb_score_mu = 188, 208
srh_score = np.random.normal(loc=srh_score_mu, scale=18, size=10000)
rcb_score = np.random.normal(loc=rcb_score_mu, scale=18, size=10000)
if srh_wins:
    margin = srh_score - rcb_score
else:
    margin = rcb_score - srh_score
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
