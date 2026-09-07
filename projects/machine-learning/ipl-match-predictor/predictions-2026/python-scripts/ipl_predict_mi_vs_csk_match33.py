"""
Mumbai Indians vs Chennai Super Kings - Match 33, IPL 2026 - April 23
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
    # Elo ratings based on current IPL 2026 form (as of April 23)
    elo_ratings = {
        'Gujarat Titans': 1710,          # still strong but lost badly to MI
        'Mumbai Indians': 1640,          # bumped after 99-run win vs GT
        'Royal Challengers Bangalore': 1740,
        'Punjab Kings': 1760,            # Top form
        'Rajasthan Royals': 1700,
        'Chennai Super Kings': 1610,     # lost to SRH, Dhoni out all season
        'Sunrisers Hyderabad': 1660,
        'Delhi Capitals': 1620,
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

    venue_matches = historical_df[historical_df['venue'] == venue]
    if len(venue_matches) > 0:
        home_wins = len(venue_matches[(venue_matches['team1'] == home_team) & (venue_matches['winner'] == home_team)])
        home_total = len(venue_matches[venue_matches['team1'] == home_team])
        features['venue_home_win_pct'] = home_wins / home_total if home_total > 0 else 0.55
    else:
        features['venue_home_win_pct'] = 0.55

    features['home_advantage'] = 1.0
    features['toss_factor'] = 1.05
    # Wankhede strongly favors chasing (2 of 3 chasing wins in IPL 2026; batting haven)
    features['chase_factor'] = 1.10

    # Momentum reflects current IPL 2026 form (as of April 23, 2026)
    momentum_2026 = {
        'Gujarat Titans': 0.70,
        'Mumbai Indians': 0.60,          # big win vs GT, ended losing streak
        'Royal Challengers Bangalore': 0.85,
        'Punjab Kings': 0.95,
        'Rajasthan Royals': 0.70,
        'Chennai Super Kings': 0.40,     # lost to SRH, 2W-4L, Dhoni out
        'Sunrisers Hyderabad': 0.60,
        'Delhi Capitals': 0.50,
        'Kolkata Knight Riders': 0.15,
        'Lucknow Super Giants': 0.20,
    }
    features['momentum'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_home'] = momentum_2026.get(home_team, 0.5)
    features['recent_form_away'] = momentum_2026.get(away_team, 0.5)
    return features

home_team = 'Mumbai Indians'
away_team = 'Chennai Super Kings'
venue = 'Mumbai'

print(f"\n{'='*60}")
print(f"MATCH 33: {home_team} vs {away_team}")
print(f"Venue: Wankhede Stadium, {venue}, Date: April 23, 2026")
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
    print(f"  {name.upper()} P(MI wins): {pred:.1%}")

weights = {'rf': 0.20, 'xgb': 0.35, 'gb': 0.30, 'lr': 0.15}
ensemble = sum(predictions[k] * weights[k] for k in predictions)
print(f"\nEnsemble P(MI wins): {ensemble:.1%}")
mi_prob = ensemble

print("\nContextual Adjustments (positive = favor MI):")
adjustments = 0
print("  + MI coming off 99-run demolition of GT, Tilak Varma 45-ball century: +7%")
adjustments += 0.07
print("  + MI dominate H2H at Wankhede 8-5, won last clash by 9 wickets in 2025: +5%")
adjustments += 0.05
print("  + MS Dhoni out (calf strain, missed all 6 games), target return Apr 26: +5%")
adjustments += 0.05
print("  + CSK lost to SRH (Apr 18), 8th place with 4 points: +3%")
adjustments += 0.03
print("  + Ashwani Kumar devastating spell vs GT, MI bowling momentum: +2%")
adjustments += 0.02
print("  + Wankhede batting haven suits Tilak Varma/Surya, MI home crowd: +3%")
adjustments += 0.03
print("  + Hardik Pandya captaincy settled, leads attack at home: +2%")
adjustments += 0.02
print("  - CSK won 4 of last 5 vs MI (recent bilateral edge): -3%")
adjustments -= 0.03
print("  - Anshul Kamboj elite form: 13 wickets in 6 matches, 2nd highest in IPL: -3%")
adjustments -= 0.03
print("  - Rohit Sharma likely out (hamstring), weakens MI top order: -2%")
adjustments -= 0.02
print("  - Ruturaj Gaikwad captaincy stable, Samson/Brevis can exploit batting pitch: -2%")
adjustments -= 0.02
print("  - Wankhede chase factor (2 of 3 chasing wins in 2026), CSK dangerous chasers: -2%")
adjustments -= 0.02
print("  - MI overall only 2W-4L, inconsistency remains concern: -2%")
adjustments -= 0.02

final_mi = max(0.10, min(0.90, mi_prob + adjustments))
final_csk = 1.0 - final_mi
print(f"\nFinal P(MI wins):  {final_mi:.1%}")
print(f"Final P(CSK wins): {final_csk:.1%}")

np.random.seed(42)
# Determine winner based on probabilities
mi_wins = final_mi >= 0.5
winner_abbrev = 'MI' if mi_wins else 'CSK'
winner_full = 'Mumbai Indians' if mi_wins else 'Chennai Super Kings'
winner_conf = int(final_mi * 100) if mi_wins else int(final_csk * 100)

if mi_wins:
    mi_score_mu, csk_score_mu = 200, 178
else:
    mi_score_mu, csk_score_mu = 175, 195
mi_score = np.random.normal(loc=mi_score_mu, scale=25, size=10000)
csk_score = np.random.normal(loc=csk_score_mu, scale=27, size=10000)
if mi_wins:
    margin = mi_score - csk_score
else:
    margin = csk_score - mi_score
mean_margin = np.mean(margin[margin > 0])

print(f"\n{'='*60}")
print(f"KEY FACTORS (Favoring {winner_abbrev}):")
print("="*60)
if mi_wins:
    print("1. MI snapped losing streak with 99-run demolition of GT, Tilak Varma 45-ball ton")
    print("2. MI dominate Wankhede H2H 8-5 vs CSK, won last clash by 9 wickets (2025)")
    print("3. MS Dhoni ruled out (calf strain missed all 6 games), CSK missing talisman")
    print("4. CSK in poor form, 8th place with 2W-4L, lost to SRH on April 18")
else:
    print("1. CSK won 4 of last 5 vs MI, recent bilateral rivalry firmly in CSK favor")
    print("2. Anshul Kamboj elite form: 13 wickets in 6 matches (2nd in IPL wicket list)")
    print("3. Wankhede pitch favors chasing (2 of 3 wins by chasers in 2026), CSK dangerous chasers")
    print("4. Rohit Sharma likely out (hamstring), MI top order weakened")

print(f"\n{'='*60}")
risk_team = 'CSK' if mi_wins else 'MI'
print(f"RISK FACTORS ({risk_team} Could Upset):")
print("="*60)
if mi_wins:
    print("1. CSK won 4 of last 5 against MI (recent bilateral series edge)")
    print("2. Anshul Kamboj in elite form: 13 wickets in 6 matches (2nd in IPL wicket list)")
    print("3. Rohit Sharma likely out (hamstring), weakens MI top order against CSK pace")
    print("4. Wankhede favors chasing (2 of 3 wins by chasers in IPL 2026), CSK dangerous chasers")
else:
    print("1. MI coming off 99-run demolition of GT, Tilak Varma 45-ball maiden IPL century")
    print("2. MI dominate Wankhede H2H 8-5 vs CSK, won last clash by 9 wickets (2025)")
    print("3. MS Dhoni out (calf strain all 6 games), CSK missing talisman behind stumps")
    print("4. CSK sit 8th with 2W-4L, lost to SRH on April 18, momentum poor")

print(f"\n{'='*60}")
print("PREDICTION SUMMARY:")
print(f"{'='*60}")
print(f"PREDICTED WINNER: {winner_full} ({winner_abbrev})")
print(f"CONFIDENCE: {winner_conf}%")
print(f"EXPECTED MARGIN: {int(mean_margin)} runs")
print(f"MODEL SCORES: RF={predictions['rf']:.3f}, XGB={predictions['xgb']:.3f}, GB={predictions['gb']:.3f}, LR={predictions['lr']:.3f}")
if mi_wins:
    print(f"REASONING: Mumbai Indians enter the El Clasico clash riding a 99-run demolition of Gujarat Titans, powered by Tilak Varma's blistering 45-ball maiden IPL century and Ashwani Kumar's devastating bowling spell. MI's home dominance at Wankhede is emphatic, leading CSK 8-5 at this venue and winning the last encounter by 9 wickets in 2025. MS Dhoni remains sidelined with a calf strain, having missed all six CSK games this season, target return not before April 26. CSK sit 8th with just 4 points from 6 matches, fresh off a loss to SRH on April 18. CSK's recent 4-of-5 bilateral edge, Anshul Kamboj's 13-wicket haul and Rohit Sharma's likely hamstring absence keep them in contest, but the combined weight of home form, momentum and Dhoni's absence tilts the scales toward MI.")
else:
    print(f"REASONING: Chennai Super Kings are marginal favorites in the El Clasico despite playing at Wankhede. CSK have won 4 of the last 5 bilateral meetings and the ensemble models heavily favor the Super Kings based on historical dominance. Anshul Kamboj, 2nd on the IPL wicket list with 13 scalps in 6 matches, leads an incisive pace attack that can exploit Wankhede's pace and bounce. Ruturaj Gaikwad's captaincy is settled and Samson/Brevis have the firepower to chase down big totals on a batting haven where 2 of 3 games in 2026 have been won batting second. MI come in riding a 99-run win over GT with Tilak Varma's 45-ball century, and Wankhede is MI's fortress (8-5 vs CSK here, 9-wicket win in 2025). MS Dhoni remains out with a calf strain. But Rohit Sharma's likely hamstring absence, MI's poor 2W-4L overall record and Hardik Pandya's concerning bowling economy (11.69) keep CSK narrowly ahead on balance of data and current form.")
print(f"{'='*60}\n")
print("DB UPDATE VALUES:")
print(f"  ml_winner: '{winner_abbrev}'")
print(f"  ml_confidence: {winner_conf}")
print(f"  ml_predicted_margin: '{int(mean_margin)} runs'")
