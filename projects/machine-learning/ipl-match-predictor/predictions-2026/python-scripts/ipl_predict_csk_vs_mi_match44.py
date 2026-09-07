"""
Chennai Super Kings vs Mumbai Indians - Match 44, IPL 2026 - May 2
ML Prediction Script using Ensemble Models
Venue: M.A. Chidambaram Stadium, Chennai (Chepauk)
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
    # Updated Elo based on IPL 2026 standings as of May 2
    # PBKS top, SRH 3rd, CSK 6th (3W-5L), MI 9th (2W-5L, 4pts)
    elo_ratings = {
        'Royal Challengers Bangalore': 1745,
        'Gujarat Titans': 1690,
        'Mumbai Indians': 1520,          # struggling badly, 2W-5L, 9th
        'Punjab Kings': 1790,            # table-topper, 13pts
        'Rajasthan Royals': 1720,        # top 4
        'Chennai Super Kings': 1640,     # 3W-5L, 6th, Chepauk fortress + Dhoni return
        'Sunrisers Hyderabad': 1730,     # 3rd, 12pts, 5-match win streak
        'Delhi Capitals': 1660,          # top 4, KL Rahul orange cap
        'Kolkata Knight Riders': 1500,
        'Lucknow Super Giants': 1490,
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
    # Chepauk: spin-friendly, low-scoring, CSK home fortress, evening dew helps chasing slightly
    features['chase_factor'] = 1.04

    momentum_2026 = {
        'Royal Challengers Bangalore': 0.60,
        'Gujarat Titans': 0.55,
        'Mumbai Indians': 0.20,          # LWLLL, only 2W from 7, heaviest defeat ever vs CSK (-103)
        'Punjab Kings': 0.95,            # table-toppers
        'Rajasthan Royals': 0.70,
        'Chennai Super Kings': 0.45,     # 3W-5L but won 5 of last 6 vs MI, Dhoni returning
        'Sunrisers Hyderabad': 0.80,
        'Delhi Capitals': 0.60,
        'Kolkata Knight Riders': 0.35,
        'Lucknow Super Giants': 0.25,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Chennai Super Kings'
away_team = 'Mumbai Indians'
venue = 'Chidambaram'

print(f"\n{'='*60}")
print(f"MATCH 44: {home_team} vs {away_team}")
print(f"Venue: M.A. Chidambaram Stadium, Chennai, Date: May 2, 2026")
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
    print(f"  {name.upper()} P(CSK wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(CSK wins): {ensemble:.1%}")
csk_prob = ensemble

print("\nContextual Adjustments (positive = favor CSK):")
adjustments = 0
print("  + Chepauk fortress - CSK home record ~70%, spin-friendly suits Jadeja/Ashwin: +5%")
adjustments += 0.05
print("  + CSK won 5 of last 6 vs MI, beat MI by 103 runs earlier in IPL 2026: +5%")
adjustments += 0.05
print("  + MS Dhoni fit to play (improving from calf injury), crowd factor massive: +4%")
adjustments += 0.04
print("  + Ravindra Jadeja elite form - 21 wickets vs MI historically, controls Chepauk spin: +3%")
adjustments += 0.03
print("  + CSK top order (Gaikwad/Conway) adapts well to Chepauk low-scoring game: +2%")
adjustments += 0.02
print("  - MI's Tilak Varma in brilliant form (101* off 45 vs GT), lone bright spot: -4%")
adjustments -= 0.04
print("  - MI desperate for wins (9th, only 4 pts), could spark unpredictable fire: -3%")
adjustments -= 0.03
print("  - MI historically 22W-19L vs CSK all-time, won 5 of last 10 at neutral/away: -2%")
adjustments -= 0.02
print("  - Rohit Sharma fitness managed but when fresh, Chepauk crowd experience helps: -1%")
adjustments -= 0.01

final_csk = max(0.10, min(0.90, csk_prob + adjustments))
final_mi = 1.0 - final_csk
print(f"\nFinal P(CSK wins):   {final_csk:.1%}")
print(f"Final P(MI wins):    {final_mi:.1%}")

np.random.seed(42)
csk_wins = final_csk >= 0.5
winner_abbrev = 'CSK' if csk_wins else 'MI'
winner_full = 'Chennai Super Kings' if csk_wins else 'Mumbai Indians'
winner_conf = int(final_csk * 100) if csk_wins else int(final_mi * 100)

if csk_wins:
    csk_score_mu, mi_score_mu = 172, 155
else:
    csk_score_mu, mi_score_mu = 158, 178
csk_score = np.random.normal(loc=csk_score_mu, scale=20, size=10000)
mi_score = np.random.normal(loc=mi_score_mu, scale=22, size=10000)
if csk_wins:
    margin = csk_score - mi_score
else:
    margin = mi_score - csk_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print(f"KEY FACTORS (Favoring {winner_abbrev}):")
print("="*60)
if csk_wins:
    print("1. Chepauk home fortress - CSK ~70% home win rate, spin turns from early overs")
    print("2. CSK beat MI by 103 runs earlier this IPL 2026 season (heaviest MI defeat ever)")
    print("3. MS Dhoni fit to return (calf injury improving), crowd factor + experience huge")
    print("4. Ravindra Jadeja elite Chepauk record - 21 wickets vs MI historically")
else:
    print("1. Tilak Varma in scintillating form (101* off 45 vs GT), MI's sole batting pillar")
    print("2. Rohit Sharma fitness managed carefully, could deliver critical innings under pressure")
    print("3. MI desperate for survival, must-win situations historically unlock peak MI")
    print("4. MI 22W-19L vs CSK all-time, competitive despite poor current form")

print(f"\n{'='*60}")
risk_team = 'MI' if csk_wins else 'CSK'
print(f"RISK FACTORS ({risk_team} Could Upset):")
print("="*60)
if csk_wins:
    print("1. Tilak Varma 101* off 45 vs GT, can single-handedly chase any Chepauk total")
    print("2. MI desperation factor - must-win situations have historically triggered MI best")
    print("3. Rohit Sharma + Hardik Pandya if fully fit = dangerous unpredictable force")
    print("4. Dew in second innings could neutralize Chepauk spin advantage for CSK")
else:
    print("1. CSK Chepauk fortress - ~70% home win rate, Jadeja + spinners control conditions")
    print("2. CSK dominated MI by 103 runs earlier this season, massive psychological edge")
    print("3. MS Dhoni returning transforms CSK's lower order + finishing completely")
    print("4. Low Chepauk scores (avg 155-165) suits CSK game plan, MI may panic chase")

print(f"\n{'='*60}")
print("PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: {winner_full} ({winner_abbrev})")
print(f"CONFIDENCE: {winner_conf}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
if csk_wins:
    print(f"REASONING: Chennai Super Kings enter Match 44 as strong favorites at the M.A. Chidambaram Stadium, Chepauk, against a struggling Mumbai Indians side. CSK's home fortress is among the most formidable in IPL history with a win rate exceeding 70%, and the spin-friendly surface perfectly suits their arsenal of Ravindra Jadeja and Matheesha Pathirana backed by Chepauk's notorious turn. The psychological edge is enormous: CSK dispatched MI by 103 runs earlier in IPL 2026, the heaviest defeat in MI's history. With MS Dhoni reportedly recovering from his calf injury and potentially returning, the Chepauk crowd factor reaches fever pitch, elevating the entire CSK squad. MI arrive with only 2 wins from 7 matches, their only recent positive being Tilak Varma's brilliant 101* off 45 balls against GT, and Rohit Sharma's fitness being carefully managed. The gulf in current form, home advantage, head-to-head dominance in 2026, and Chepauk's spinning conditions all point strongly to a CSK victory, though Tilak Varma and MI's desperation factor keep the contest from being a foregone conclusion.")
else:
    print(f"REASONING: Mumbai Indians pull off an upset at Chepauk in a match they had to win. Tilak Varma's extraordinary form carried MI through a difficult chase on a spin-friendly surface while CSK's bowlers failed to defend the target. Rohit Sharma contributed a crucial partnership and MI's desperation - having only 2 wins from 7 - triggered the kind of inspired performance they are historically capable of in must-win situations.")
print(f"{'='*60}\n")
print("DB UPDATE VALUES:")
print(f"  ml_winner: '{winner_abbrev}'")
print(f"  ml_confidence: {winner_conf}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
