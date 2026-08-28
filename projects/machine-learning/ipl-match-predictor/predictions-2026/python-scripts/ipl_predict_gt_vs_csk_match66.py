"""
Gujarat Titans vs Chennai Super Kings - Match 66, IPL 2026 - May 21 (7:30 PM IST)
ML Prediction Script using Ensemble Models
Venue: Narendra Modi Stadium, Ahmedabad (GT home)

Context (web-researched 2026-05-21):
  GT 12P 8W-4L = 16 pts, #1 in points table, NRR 0.551, ALREADY QUALIFIED for playoffs.
    Hot form: 5-match winning streak. Crushed SRH by 82 runs at home in M56.
    Captain: Shubman Gill (552 runs, #2 Orange Cap, SR 160.46).
    Top scorer: B Sai Sudharsan (553 runs, #1 ORANGE CAP, SR 157.54).
    Wicketkeeper-bat: Jos Buttler in form.
    Bowling: Rashid Khan (16 wkts, joint #4 Purple Cap), Kagiso Rabada (21 wkts, #2 Purple Cap),
            Mohammed Siraj, Prasidh Krishna - one of strongest seam attacks in tournament.
    Playing for top-2 seeding (Qualifier 1 advantage).

  CSK 12P 5W-7L (approx) = ~10 pts, 6th place, playoff hopes hanging by a thread.
    Cold form: Lost to SRH at home Chepauk May 18 (M63), playoff destiny no longer in own hands.
    Captain: Ruturaj Gaikwad.
    Top scorers: Dewald Brevis (44 vs SRH), Kartik Sharma (32, also 54* vs MI), Shivam Dube (26 vs SRH).
    Wicketkeeper: Sanju Samson.
    Bowling: Anshul Kamboj (19 wkts, #3 Purple Cap), Noor Ahmad (spin), Spencer Johnson (pace).
    Mathematical playoff hope but needs to win + other results.

  H2H all-time: GT 4 - CSK 4 (8 matches, balanced 50% each).
  H2H recent: CSK won last 3 fixtures including 83-run demolition at Ahmedabad on May 25, 2025
              (CSK 230, GT 147 a/o). CSK is GT's bogey side recently despite GT's overall season form.

  Narendra Modi Stadium IPL 2026:
    - 1st innings avg ~181 in 2026 (down from 220+ in 2025)
    - Bat-first wins ~60% per AllCric stats
    - BUT evening dew significant in May 37C+ days -> ball slippery, chasing advantage from over 12
    - Mixed signal - toss winner likely bowls to use dew
    - Avg first innings score historically ~178
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
    # Updated Elo as of May 21 morning (post-M65 KKR vs MI)
    elo_ratings = {
        'Royal Challengers Bangalore': 1830,
        'Gujarat Titans': 1825,            # +10 bumped from M65 day; #1 table, 5-match streak
        'Sunrisers Hyderabad': 1755,        # +10 won at Chepauk vs CSK M63
        'Chennai Super Kings': 1695,        # -15 lost M63 at home, sliding to 6th
        'Punjab Kings': 1685,
        'Delhi Capitals': 1665,
        'Kolkata Knight Riders': 1655,
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

    features['home_advantage'] = 1.08   # NMS Ahmedabad - GT's true home, sellout crowd vs CSK
    features['toss_factor'] = 1.05
    features['chase_factor'] = 1.05     # mild dew benefit at NMS for chasers in May evening

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.85,
        'Gujarat Titans': 0.88,            # 5 wins in a row, crushed SRH by 82
        'Sunrisers Hyderabad': 0.72,
        'Chennai Super Kings': 0.40,       # cold, lost to SRH at home, slipping
        'Punjab Kings': 0.22,
        'Mumbai Indians': 0.52,
        'Delhi Capitals': 0.55,
        'Rajasthan Royals': 0.30,
        'Kolkata Knight Riders': 0.75,
        'Lucknow Super Giants': 0.22,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Gujarat Titans'
away_team = 'Chennai Super Kings'
venue = 'Narendra Modi Stadium'

print(f"\n{'='*60}")
print(f"MATCH 66: {home_team} vs {away_team}")
print(f"Venue: Narendra Modi Stadium, Ahmedabad")
print(f"Time: 7:30 PM IST (evening, dew factor moderate)")
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
    print(f"  {name.upper()} P(GT wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(GT wins): {ensemble:.1%}")
gt_prob = ensemble

print("\nContextual Adjustments (positive = favor GT):")
adjustments = 0
print("  + GT #1 in table 16 pts vs CSK 6th ~10 pts, biggest class gap of the week: +6%")
adjustments += 0.06
print("  + GT 5-match winning streak, just crushed SRH by 82 runs at this exact venue: +5%")
adjustments += 0.05
print("  + GT home Narendra Modi Stadium sellout, playing for top-2 seeding (Q1 advantage): +4%")
adjustments += 0.04
print("  + Sai Sudharsan Orange Cap (553 runs) + Shubman Gill #2 (552 runs) = elite top order: +4%")
adjustments += 0.04
print("  + Rabada 21 wkts + Rashid + Siraj seam-spin combo elite vs CSK middle order: +3%")
adjustments += 0.03
print("  + CSK lost to SRH at Chepauk (home) on May 18, playoff destiny no longer in their hands: +3%")
adjustments += 0.03
print("  - CSK have won last 3 H2H including 83-run demolition of GT at THIS venue May 2025: -5%")
adjustments -= 0.05
print("  - Anshul Kamboj #3 Purple Cap 19 wkts gives CSK a wicket-taking threat in powerplay: -3%")
adjustments -= 0.03
print("  - CSK in win-or-out mode - desperation factor + Ruturaj Gaikwad capable of clutch knock: -2%")
adjustments -= 0.02
print("  - Noor Ahmad wrist-spin + Spencer Johnson left-arm pace can trouble GT right-handed top order: -2%")
adjustments -= 0.02

final_gt = max(0.10, min(0.90, gt_prob + adjustments))
final_csk = 1.0 - final_gt
print(f"\nFinal P(GT wins): {final_gt:.1%}")
print(f"Final P(CSK wins): {final_csk:.1%}")

np.random.seed(42)
gt_wins = final_gt >= 0.5
winner_abbrev = 'GT' if gt_wins else 'CSK'
winner_full = 'Gujarat Titans' if gt_wins else 'Chennai Super Kings'
winner_conf = int(final_gt * 100) if gt_wins else int(final_csk * 100)

# Narendra Modi Stadium IPL 2026: 1st inn avg ~181, par 175-190, mild chase edge
if gt_wins:
    gt_score_mu, csk_score_mu = 195, 173
else:
    gt_score_mu, csk_score_mu = 168, 195
gt_score = np.random.normal(loc=gt_score_mu, scale=18, size=10000)
csk_score = np.random.normal(loc=csk_score_mu, scale=18, size=10000)
if gt_wins:
    margin = gt_score - csk_score
else:
    margin = csk_score - gt_score
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
