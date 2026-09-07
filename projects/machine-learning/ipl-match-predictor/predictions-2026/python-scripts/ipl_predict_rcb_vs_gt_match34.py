"""
Royal Challengers Bengaluru vs Gujarat Titans - Match 34, IPL 2026 - April 24
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
# Normalize RCB name variations to historical spelling used for training
df['team1'] = df['team1'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['team2'] = df['team2'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
df['winner'] = df['winner'].replace({'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'})
print(f"Loaded {len(df)} historical matches (2008-2024)")

def engineer_features(home_team, away_team, venue, historical_df):
    features = {}
    # Elo ratings based on current IPL 2026 form (as of April 24, Match 34)
    elo_ratings = {
        'Royal Challengers Bangalore': 1755,  # Defending champs, 4W-2L, 3rd
        'Gujarat Titans': 1685,                # 3W-3L 6th, hammered by MI (99 runs)
        'Mumbai Indians': 1670,                # bumped after 99-run win vs GT
        'Punjab Kings': 1760,                  # Top form
        'Rajasthan Royals': 1700,
        'Chennai Super Kings': 1600,           # lost to MI, Dhoni out all season
        'Sunrisers Hyderabad': 1660,
        'Delhi Capitals': 1640,                # beat RCB
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
    # Chinnaswamy heavily favors chasing with dew in the second innings
    features['chase_factor'] = 1.10

    # Momentum reflects current IPL 2026 form (as of April 24, 2026)
    momentum_2026 = {
        'Royal Challengers Bangalore': 0.70,   # 4W-2L, lost last to DC
        'Gujarat Titans': 0.50,                # 3W-3L, coming off 99-run loss vs MI
        'Mumbai Indians': 0.65,                # back on track after GT demolition
        'Punjab Kings': 0.95,
        'Rajasthan Royals': 0.70,
        'Chennai Super Kings': 0.35,
        'Sunrisers Hyderabad': 0.55,
        'Delhi Capitals': 0.60,
        'Kolkata Knight Riders': 0.15,
        'Lucknow Super Giants': 0.20,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Royal Challengers Bangalore'
away_team = 'Gujarat Titans'
venue = 'M Chinnaswamy'

print(f"\n{'='*60}")
print(f"MATCH 34: {home_team} vs {away_team}")
print(f"Venue: M Chinnaswamy Stadium, Bengaluru, Date: April 24, 2026")
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
print("  + RCB defending champions, retained 17 from 2025 title-winning squad: +5%")
adjustments += 0.05
print("  + Chinnaswamy is RCB's fortress, Kohli averages 52+ at this venue: +5%")
adjustments += 0.05
print("  + Patidar in prime captain form, 222 runs (3rd-highest IPL 2026): +3%")
adjustments += 0.03
print("  + Hazlewood/Bhuvneshwar new-ball threat exploits first 3 overs movement: +3%")
adjustments += 0.03
print("  + Phil Salt and Kohli explosive PowerPlay duo at batting paradise: +3%")
adjustments += 0.03
print("  + RCB 4W-2L (3rd), proven at chasing under lights with dew: +2%")
adjustments += 0.02
print("  + Tim David finishing role, Krunal Pandya all-round depth: +2%")
adjustments += 0.02
print("  - Shubman Gill in elite form: 235 runs in 5 innings, leading run-scorer: -4%")
adjustments -= 0.04
print("  - Prasidh Krishna leading IPL wicket-taker with 11 scalps: -3%")
adjustments -= 0.03
print("  - GT spin trio (Rashid, Sai Kishore, Sundar) effective on flat decks: -3%")
adjustments -= 0.03
print("  - GT desperate to bounce back after 99-run loss vs MI, character test: -2%")
adjustments -= 0.02
print("  - Buttler/Sudharsan top order can post 200+ chasing under dew: -2%")
adjustments -= 0.02
print("  - RCB lost last match to DC by 6 wickets, brief stumble in form: -2%")
adjustments -= 0.02

final_rcb = max(0.10, min(0.90, rcb_prob + adjustments))
final_gt = 1.0 - final_rcb
print(f"\nFinal P(RCB wins):  {final_rcb:.1%}")
print(f"Final P(GT wins):   {final_gt:.1%}")

np.random.seed(42)
rcb_wins = final_rcb >= 0.5
winner_abbrev = 'RCB' if rcb_wins else 'GT'
winner_full = 'Royal Challengers Bengaluru' if rcb_wins else 'Gujarat Titans'
winner_conf = int(final_rcb * 100) if rcb_wins else int(final_gt * 100)

if rcb_wins:
    rcb_score_mu, gt_score_mu = 205, 185
else:
    rcb_score_mu, gt_score_mu = 180, 200
rcb_score = np.random.normal(loc=rcb_score_mu, scale=25, size=10000)
gt_score = np.random.normal(loc=gt_score_mu, scale=27, size=10000)
if rcb_wins:
    margin = rcb_score - gt_score
else:
    margin = gt_score - rcb_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print(f"KEY FACTORS (Favoring {winner_abbrev}):")
print("="*60)
if rcb_wins:
    print("1. Defending champions with 17 retained from 2025 title squad, settled core")
    print("2. Chinnaswamy fortress advantage, Kohli averages 52+ and Patidar 222 runs this season")
    print("3. Hazlewood-Bhuvneshwar new-ball attack exploits first 3 overs seam movement")
    print("4. Phil Salt + Kohli PowerPlay duo on batting paradise, plus dew-chase edge")
else:
    print("1. Shubman Gill elite form: 235 runs in 5 innings, leading IPL run-scorer")
    print("2. Prasidh Krishna leads IPL wicket list with 11 scalps in 6 matches")
    print("3. GT spin trio (Rashid, Sai Kishore, Sundar) grip flat Chinnaswamy surface")
    print("4. Buttler-Sudharsan support can post 200+ under dew, GT desperate to respond")

print(f"\n{'='*60}")
risk_team = 'GT' if rcb_wins else 'RCB'
print(f"RISK FACTORS ({risk_team} Could Upset):")
print("="*60)
if rcb_wins:
    print("1. Shubman Gill in elite form (235 runs in 5 innings, leading IPL run-scorer)")
    print("2. Prasidh Krishna tops IPL wicket list (11 wickets, can rip through top order)")
    print("3. GT spin trio (Rashid-Sai Kishore-Sundar) effective on flat 200-type decks")
    print("4. GT will be desperate to bounce back from 99-run hammering by MI")
else:
    print("1. RCB defending champs, 17 retained from title squad, Chinnaswamy fortress")
    print("2. Patidar captain form (222 runs) + Kohli 52+ average at this venue")
    print("3. Hazlewood-Bhuvneshwar new-ball pair exploit seam movement in first 3 overs")
    print("4. Phil Salt-Kohli PowerPlay pair + Tim David finisher role on batting paradise")

print(f"\n{'='*60}")
print("PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: {winner_full} ({winner_abbrev})")
print(f"CONFIDENCE: {winner_conf}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
if rcb_wins:
    print(f"REASONING: Royal Challengers Bengaluru enter Match 34 as narrow favorites at their Chinnaswamy fortress. The defending champions retained 17 players from their 2025 title-winning squad, ensuring a settled core led by captain Rajat Patidar who sits 3rd in the IPL 2026 run charts with 222 runs. Virat Kohli averages over 52 at this venue across his career and opens alongside the explosive Phil Salt, a PowerPlay pairing perfectly suited to the batting paradise. Josh Hazlewood and Bhuvneshwar Kumar provide the new-ball threat needed to exploit the first three overs of seam movement before the deck flattens. Gujarat Titans arrive in distress after a 99-run capitulation against Mumbai Indians and sit 6th, though Shubman Gill leads the IPL run charts (235 in 5 innings) and Prasidh Krishna tops the wicket list (11 scalps). GT's spin trio of Rashid Khan, Sai Kishore and Washington Sundar can trouble RCB on a flat deck, and Jos Buttler provides firepower at the top. But the combined weight of home advantage, champion continuity, and the dew-chase factor tilts the scales toward the hosts.")
else:
    print(f"REASONING: Gujarat Titans are narrow favorites despite arriving at Chinnaswamy after a 99-run loss to Mumbai Indians. Shubman Gill leads the IPL 2026 run charts with 235 runs in 5 innings, while Prasidh Krishna tops the wicket list with 11 scalps in 6 matches. The GT spin trio of Rashid Khan, Sai Kishore and Washington Sundar is ideally suited to flat 200-type decks, and Jos Buttler provides the explosive top-order firepower to chase big totals under dew. RCB counter with their defending-champions DNA at a fortress venue, Patidar's 222-run lead in form, and Kohli's 52-plus Chinnaswamy average, plus the Hazlewood-Bhuvneshwar new-ball threat in the first three overs. But GT's individual brilliance at the top and wickets column, combined with the desperation to respond from a heavy loss, keep them narrowly ahead on aggregate.")
print(f"{'='*60}\n")
print("DB UPDATE VALUES:")
print(f"  ml_winner: '{winner_abbrev}'")
print(f"  ml_confidence: {winner_conf}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
