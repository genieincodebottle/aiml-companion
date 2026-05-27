"""
IPL 2026 Match Prediction: LSG vs GT (Match 19, April 12, 2026)
================================================================
BRSABV Ekana Cricket Stadium, Lucknow | 3:30 PM IST
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_DIR = r"c:\projects\aiml-companion\projects\ipl-match-predictor\data\raw"

print("=" * 70)
print("  IPL 2026 MATCH PREDICTION: LSG vs GT")
print("  Match 19 | BRSABV Ekana Cricket Stadium, Lucknow")
print("  April 12, 2026 | 3:30 PM IST")
print("=" * 70)

print("\n[1/7] Loading historical data (2008-2024)...")
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"), na_values=["NA", ""])
deliveries = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"), na_values=["NA", ""])

NAME_MAP = {
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}
for col in ["team1", "team2", "toss_winner", "winner"]:
    matches[col] = matches[col].replace(NAME_MAP)

matches["winner"] = matches["winner"].fillna("No Result")
matches["city"] = matches["city"].fillna("Unknown")
matches["result_margin"] = matches["result_margin"].fillna(0)
matches["toss_decision"] = matches["toss_decision"].fillna("field")
matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d", errors="coerce")
matches = matches.sort_values("date").reset_index(drop=True)
print(f"  Loaded {len(matches)} historical matches")

# ============================================================================
# 2. SUPPLEMENT WITH IPL 2025 + 2026 DATA
# ============================================================================
print("\n[2/7] Supplementing with IPL 2025 + 2026 season data...")

ipl_2025_plus_matches = [
    # LSG 2025: 6W 8L, 7th place - missed playoffs
    {"date": "2025-03-25", "team1": "Lucknow Super Giants", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-01", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-07", "team1": "Lucknow Super Giants", "team2": "Mumbai Indians", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-13", "team1": "Lucknow Super Giants", "team2": "Kolkata Knight Riders", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-04-19", "team1": "Gujarat Titans", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-25", "team1": "Lucknow Super Giants", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-01", "team1": "Sunrisers Hyderabad", "team2": "Lucknow Super Giants", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-07", "team1": "Lucknow Super Giants", "team2": "Chennai Super Kings", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2025-05-13", "team1": "Royal Challengers Bangalore", "team2": "Lucknow Super Giants", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-05-19", "team1": "Lucknow Super Giants", "team2": "Gujarat Titans", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-25", "team1": "Punjab Kings", "team2": "Lucknow Super Giants", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-05-30", "team1": "Lucknow Super Giants", "team2": "Sunrisers Hyderabad", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-06-01", "team1": "Kolkata Knight Riders", "team2": "Lucknow Super Giants", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-06-02", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},

    # GT 2025: 9W 5L, 3rd place - lost in Eliminator to MI
    {"date": "2025-03-26", "team1": "Gujarat Titans", "team2": "Rajasthan Royals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-02", "team1": "Royal Challengers Bangalore", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-08", "team1": "Gujarat Titans", "team2": "Mumbai Indians", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-04-14", "team1": "Punjab Kings", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-19", "team1": "Gujarat Titans", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-25", "team1": "Gujarat Titans", "team2": "Delhi Capitals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-05-01", "team1": "Chennai Super Kings", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-07", "team1": "Gujarat Titans", "team2": "Sunrisers Hyderabad", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-05-13", "team1": "Kolkata Knight Riders", "team2": "Gujarat Titans", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-05-19", "team1": "Lucknow Super Giants", "team2": "Gujarat Titans", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-25", "team1": "Gujarat Titans", "team2": "Rajasthan Royals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-05-30", "team1": "Gujarat Titans", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "wickets", "result_margin": 3},
    {"date": "2025-06-01", "team1": "Mumbai Indians", "team2": "Gujarat Titans", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-06-02", "team1": "Chennai Super Kings", "team2": "Gujarat Titans", "winner": "Chennai Super Kings", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "bat", "result": "runs", "result_margin": 50},

    # IPL 2026 results so far (Matches 1-18)
    {"date": "2026-03-28", "team1": "Royal Challengers Bangalore", "team2": "Sunrisers Hyderabad", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2026-03-29", "team1": "Punjab Kings", "team2": "Gujarat Titans", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2026-04-01", "team1": "Mumbai Indians", "team2": "Kolkata Knight Riders", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 17},
    {"date": "2026-04-02", "team1": "Rajasthan Royals", "team2": "Chennai Super Kings", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 17},
    {"date": "2026-04-01", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-02", "team1": "Sunrisers Hyderabad", "team2": "Kolkata Knight Riders", "winner": "Sunrisers Hyderabad", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 65},
    {"date": "2026-04-03", "team1": "Rajasthan Royals", "team2": "Gujarat Titans", "winner": "Rajasthan Royals", "city": "Guwahati", "venue": "Barsapara Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2026-04-04", "team1": "Chennai Super Kings", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2026-04-04", "team1": "Delhi Capitals", "team2": "Mumbai Indians", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-05", "team1": "Rajasthan Royals", "team2": "Delhi Capitals", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2026-04-05", "team1": "Sunrisers Hyderabad", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2026-04-05", "team1": "Royal Challengers Bangalore", "team2": "Chennai Super Kings", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "runs", "result_margin": 43},
    {"date": "2026-04-06", "team1": "Punjab Kings", "team2": "Kolkata Knight Riders", "winner": "No Result", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "no result", "result_margin": 0},
    {"date": "2026-04-07", "team1": "Rajasthan Royals", "team2": "Mumbai Indians", "winner": "Rajasthan Royals", "city": "Guwahati", "venue": "Barsapara Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 27},
    {"date": "2026-04-08", "team1": "Gujarat Titans", "team2": "Delhi Capitals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "runs", "result_margin": 1},
    {"date": "2026-04-08", "team1": "Sunrisers Hyderabad", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2026-04-09", "team1": "Kolkata Knight Riders", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 3},
    {"date": "2026-04-10", "team1": "Rajasthan Royals", "team2": "Royal Challengers Bangalore", "winner": "Rajasthan Royals", "city": "Guwahati", "venue": "Barsapara Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2026-04-11", "team1": "Punjab Kings", "team2": "Sunrisers Hyderabad", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-11", "team1": "Chennai Super Kings", "team2": "Delhi Capitals", "winner": "Chennai Super Kings", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "bat", "result": "runs", "result_margin": 23},
]

supp_df = pd.DataFrame(ipl_2025_plus_matches)
supp_df["date"] = pd.to_datetime(supp_df["date"])
supp_df["season"] = supp_df["date"].dt.year.astype(str)
supp_df["id"] = range(900000, 900000 + len(supp_df))
supp_df["match_type"] = "League"
supp_df["super_over"] = "N"
supp_df["player_of_match"] = "Unknown"

all_matches = pd.concat([matches, supp_df], ignore_index=True)
all_matches = all_matches.sort_values("date").reset_index(drop=True)
print(f"  Added {len(supp_df)} supplemental matches -> Total: {len(all_matches)} matches")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n[3/7] Engineering features...")

INITIAL_ELO = 1500
K_FACTOR = 32

def compute_elo(df):
    ratings = defaultdict(lambda: INITIAL_ELO)
    elo_t1, elo_t2 = [], []
    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        r1, r2 = ratings[t1], ratings[t2]
        elo_t1.append(r1)
        elo_t2.append(r2)
        winner = row.get("winner", None)
        if pd.isna(winner) or winner in ("No Result", "Unknown"):
            s1 = 0.5
        elif winner == t1:
            s1 = 1.0
        else:
            s1 = 0.0
        e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        ratings[t1] = r1 + K_FACTOR * (s1 - e1)
        ratings[t2] = r2 + K_FACTOR * ((1 - s1) - (1 - e1))
    df = df.copy()
    df["elo_team1"] = elo_t1
    df["elo_team2"] = elo_t2
    df["elo_diff"] = df["elo_team1"] - df["elo_team2"]
    return df, ratings

all_matches, final_elo = compute_elo(all_matches)

def compute_momentum(df, window=5):
    team_results = defaultdict(list)
    m1_list, m2_list = [], []
    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        winner = row.get("winner", None)
        recent_t1 = team_results[t1][-window:]
        recent_t2 = team_results[t2][-window:]
        m1 = sum(recent_t1) / len(recent_t1) if recent_t1 else 0.5
        m2 = sum(recent_t2) / len(recent_t2) if recent_t2 else 0.5
        m1_list.append(m1)
        m2_list.append(m2)
        if not pd.isna(winner) and winner not in ("No Result", "Unknown"):
            team_results[t1].append(1 if winner == t1 else 0)
            team_results[t2].append(1 if winner == t2 else 0)
    df = df.copy()
    df["momentum_team1"] = m1_list
    df["momentum_team2"] = m2_list
    df["momentum_diff"] = df["momentum_team1"] - df["momentum_team2"]
    return df

all_matches = compute_momentum(all_matches)

def compute_h2h(df):
    h2h = defaultdict(lambda: {"wins": defaultdict(int), "total": 0})
    h2h_wr, h2h_cnt = [], []
    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        key = tuple(sorted([t1, t2]))
        winner = row.get("winner", None)
        total = h2h[key]["total"]
        if total > 0:
            h2h_wr.append(h2h[key]["wins"][t1] / total)
        else:
            h2h_wr.append(0.5)
        h2h_cnt.append(total)
        if not pd.isna(winner) and winner not in ("No Result", "Unknown"):
            h2h[key]["total"] += 1
            h2h[key]["wins"][winner] += 1
    df = df.copy()
    df["h2h_team1_winrate"] = h2h_wr
    df["h2h_matches"] = h2h_cnt
    return df

all_matches = compute_h2h(all_matches)

HOME_CITIES = {
    "Mumbai Indians": "Mumbai",
    "Chennai Super Kings": "Chennai",
    "Royal Challengers Bangalore": "Bangalore",
    "Kolkata Knight Riders": "Kolkata",
    "Delhi Capitals": "Delhi",
    "Rajasthan Royals": "Jaipur",
    "Sunrisers Hyderabad": "Hyderabad",
    "Punjab Kings": "Mohali",
    "Lucknow Super Giants": "Lucknow",
    "Gujarat Titans": "Ahmedabad",
}

all_matches["home_team1"] = all_matches.apply(
    lambda r: 1 if HOME_CITIES.get(r["team1"], "") == r.get("city", "") else 0, axis=1
)
all_matches["home_team2"] = all_matches.apply(
    lambda r: 1 if HOME_CITIES.get(r["team2"], "") == r.get("city", "") else 0, axis=1
)

venue_chase_bias = {
    "Eden Gardens": 56 / 97,
    "M Chinnaswamy Stadium": 53 / 98,
    "Wankhede Stadium": 0.52,
    "MA Chidambaram Stadium": 0.45,
    "Rajiv Gandhi International Stadium": 0.50,
    "Narendra Modi Stadium": 0.47,
    "Arun Jaitley Stadium": 0.51,
    "BRSABV Ekana Cricket Stadium": 11 / 20,
    "Maharaja Yadavindra Singh International Cricket Stadium": 7 / 17,
    "Sawai Mansingh Stadium": 0.50,
    "Barsapara Cricket Stadium": 0.60,
}

all_matches["venue_chase_bias"] = all_matches["venue"].map(venue_chase_bias).fillna(0.50)
all_matches["toss_chose_field"] = (all_matches["toss_decision"] == "field").astype(int)
all_matches["toss_winner_is_team1"] = (all_matches["toss_winner"] == all_matches["team1"]).astype(int)
all_matches["season_year"] = all_matches["date"].dt.year
all_matches["recency_weight"] = (all_matches["season_year"] - 2008 + 1) / (2026 - 2008 + 1)
all_matches["elo_x_momentum_t1"] = all_matches["elo_team1"] * all_matches["momentum_team1"]
all_matches["elo_x_momentum_t2"] = all_matches["elo_team2"] * all_matches["momentum_team2"]
all_matches["elo_x_home_t1"] = all_matches["elo_team1"] * all_matches["home_team1"]

n_features = len([c for c in all_matches.columns if c.startswith(("elo_", "momentum_", "h2h_", "home_", "venue_", "toss_", "recency_"))])
print(f"  Engineered {n_features} features")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n[4/7] Training ensemble models...")

all_matches["team1_won"] = (all_matches["winner"] == all_matches["team1"]).astype(int)
valid = all_matches.dropna(subset=["winner"])
valid = valid[~valid["winner"].isin(["No Result", "Unknown"])]

FEATURE_COLS = [
    "elo_team1", "elo_team2", "elo_diff",
    "momentum_team1", "momentum_team2", "momentum_diff",
    "h2h_team1_winrate", "h2h_matches",
    "home_team1", "home_team2",
    "venue_chase_bias", "toss_chose_field", "toss_winner_is_team1",
    "recency_weight",
    "elo_x_momentum_t1", "elo_x_momentum_t2", "elo_x_home_t1",
]

X = valid[FEATURE_COLS].fillna(0)
y = valid["team1_won"]
sample_weights = valid["recency_weight"].values

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS, index=X.index)

rf = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=15, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    cv=5, method="isotonic"
)
rf.fit(X_scaled, y, sample_weight=sample_weights)
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"), X_scaled, y, cv=5, scoring="accuracy")
print(f"  Random Forest CV: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

if HAS_XGB:
    xgb = CalibratedClassifierCV(
        XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, random_state=42, eval_metric="logloss"),
        cv=5, method="isotonic"
    )
    xgb.fit(X_scaled, y, sample_weight=sample_weights)
    xgb_cv = cross_val_score(XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss"), X_scaled, y, cv=5, scoring="accuracy")
    print(f"  XGBoost CV:       {xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f})")

gb = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_split=15, min_samples_leaf=8, random_state=42),
    cv=5, method="isotonic"
)
gb.fit(X_scaled, y, sample_weight=sample_weights)
gb_cv = cross_val_score(GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42), X_scaled, y, cv=5, scoring="accuracy")
print(f"  Gradient Boost CV: {gb_cv.mean():.4f} (+/- {gb_cv.std():.4f})")

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_scaled, y, sample_weight=sample_weights)
lr_cv = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy")
print(f"  Logistic Reg CV:  {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")

# ============================================================================
# 5. PREDICT: LSG vs GT
# ============================================================================
print("\n[5/7] Predicting LSG vs GT...")

LSG = "Lucknow Super Giants"
GT = "Gujarat Titans"

lsg_elo = final_elo[LSG]
gt_elo = final_elo[GT]

lsg_matches = all_matches[(all_matches["team1"] == LSG) | (all_matches["team2"] == LSG)].tail(5)
gt_matches = all_matches[(all_matches["team1"] == GT) | (all_matches["team2"] == GT)].tail(5)

lsg_wins_last5 = sum(lsg_matches["winner"] == LSG)
gt_wins_last5 = sum(gt_matches["winner"] == GT)
lsg_momentum = lsg_wins_last5 / 5
gt_momentum = gt_wins_last5 / 5

lsg_gt = all_matches[
    ((all_matches["team1"] == LSG) & (all_matches["team2"] == GT)) |
    ((all_matches["team1"] == GT) & (all_matches["team2"] == LSG))
]
total_h2h = len(lsg_gt[lsg_gt["winner"].isin([LSG, GT])])
lsg_h2h_wins = sum(lsg_gt["winner"] == LSG)
gt_h2h_wins = total_h2h - lsg_h2h_wins
lsg_h2h_winrate = lsg_h2h_wins / total_h2h if total_h2h > 0 else 0.5

# LSG is team1 (HOME - Lucknow), GT is team2
match_features = {
    "elo_team1": lsg_elo,
    "elo_team2": gt_elo,
    "elo_diff": lsg_elo - gt_elo,
    "momentum_team1": lsg_momentum,
    "momentum_team2": gt_momentum,
    "momentum_diff": lsg_momentum - gt_momentum,
    "h2h_team1_winrate": lsg_h2h_winrate,
    "h2h_matches": total_h2h,
    "home_team1": 1,   # LSG home ground - Ekana Stadium
    "home_team2": 0,
    "venue_chase_bias": 11 / 20,   # Ekana - slight chase advantage
    "toss_chose_field": 1,  # Afternoon match but dew still factor
    "toss_winner_is_team1": 0.5,
    "recency_weight": 1.0,
    "elo_x_momentum_t1": lsg_elo * lsg_momentum,
    "elo_x_momentum_t2": gt_elo * gt_momentum,
    "elo_x_home_t1": lsg_elo * 1,
}

X_pred = pd.DataFrame([match_features])
X_pred_scaled = pd.DataFrame(scaler.transform(X_pred), columns=FEATURE_COLS)

rf_prob = rf.predict_proba(X_pred_scaled)[0]
gb_prob = gb.predict_proba(X_pred_scaled)[0]
lr_prob = lr.predict_proba(X_pred_scaled)[0]

if HAS_XGB:
    xgb_prob = xgb.predict_proba(X_pred_scaled)[0]
    ensemble_prob = 0.20 * rf_prob + 0.35 * xgb_prob + 0.30 * gb_prob + 0.15 * lr_prob
else:
    ensemble_prob = 0.30 * rf_prob + 0.45 * gb_prob + 0.25 * lr_prob

# ============================================================================
# 6. CONTEXTUAL ADJUSTMENTS
# ============================================================================
print("\n[6/7] Applying contextual adjustments...")

base_lsg_prob = ensemble_prob[1]  # P(LSG wins)

adjustments = {}

# A. LSG 2W-1L in 2026 - good away form (+2%)
# Beat SRH away (5 wkts), Beat KKR away (dramatic 3 wkts - Mukul Choudhary 54 off 27)
# Only loss was to DC in Match 5 (bowled out for 141)
# Strong momentum after two consecutive away wins
adjustments["lsg_good_2026_form"] = +0.02

# B. LSG home advantage at Ekana (+2%)
# Ekana is LSG's fortress - long boundaries (81m straight) suit their bowling
# Crowd factor significant
# BUT lost their only home game this season (vs DC, bowled out 141)
# Reduced from +3% to +2% due to home loss concern
adjustments["lsg_ekana_home"] = +0.02

# C. LSG won last 3 H2H vs GT (since Apr 2024) (+2%)
# Completely reversed 0-4 deficit
# Won both 2025 matches vs GT convincingly
# Psychological edge in this fixture now
adjustments["lsg_recent_h2h_dominance"] = +0.02

# D. Rishabh Pant captaincy + Nicholas Pooran power (+1%)
# Pant's attacking captaincy has been sharp
# Pooran provides explosive middle-order power
# Mukul Choudhary emerging as match-winner (54 off 27 vs KKR)
adjustments["lsg_pant_pooran_mukul"] = +0.01

# E. Mohammed Shami pace at Ekana (+1%)
# Shami's seam movement with new ball dangerous
# Ekana's pace-friendly surface suits him
# Avesh Khan provides backup pace
adjustments["lsg_shami_pace"] = +0.01

# F. GT 1W-2L in 2026 - inconsistent (-1% for GT = +1% for LSG)
# Lost to PBKS (20 runs), Lost to RR (18 runs)
# Only win: thrilling 1-run win vs DC (Match 14)
# Buttler (50) and Gill (70 off 45) showed form vs DC
# But defense of 210 was nervy - needed last-ball run-out
adjustments["gt_inconsistent_2026"] = +0.01

# G. GT batting firepower - Gill, Buttler, Sudharsan (-3%)
# Shubman Gill regaining form (70 off 45 vs DC)
# Jos Buttler (116 runs in 3 innings, 50 vs DC) - dangerous
# Sai Sudharsan (Orange Cap 2025, 759 runs) - consistent
# These 3 can destroy any bowling attack
adjustments["gt_batting_firepower"] = -0.03

# H. Rashid Khan + Prasidh Krishna bowling (-2%)
# Rashid Khan (9 wkts in LSG vs GT H2H) - best record in this fixture
# Prasidh Krishna (Purple Cap 2025, 25 wkts) - lethal pace
# Washington Sundar provides spin depth
adjustments["gt_rashid_prasidh_bowling"] = -0.02

# I. 3:30 PM afternoon match - less dew (+1% for LSG batting first)
# Afternoon matches have less dew factor
# Reduces chasing team's advantage
# Slight edge if LSG win toss and bat
adjustments["afternoon_less_dew"] = +0.01

total_adjustment = sum(adjustments.values())
adjusted_lsg_prob = np.clip(base_lsg_prob + total_adjustment, 0.05, 0.95)
adjusted_gt_prob = 1 - adjusted_lsg_prob

# ============================================================================
# 7. MONTE CARLO SIMULATION
# ============================================================================
print("\n[7/7] Running Monte Carlo simulation (10,000 matches)...")

np.random.seed(42)
N_SIM = 10000

lsg_wins_sim = 0
gt_wins_sim = 0
lsg_margins = []
gt_margins = []

for _ in range(N_SIM):
    noise = np.random.normal(0, 0.05)
    match_prob = np.clip(adjusted_lsg_prob + noise, 0.1, 0.9)
    if np.random.random() < match_prob:
        lsg_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        lsg_margins.append(margin)
    else:
        gt_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        gt_margins.append(margin)

sim_lsg_pct = lsg_wins_sim / N_SIM * 100
sim_gt_pct = gt_wins_sim / N_SIM * 100

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  PREDICTION REPORT: LSG vs GT | IPL 2026 Match 19")
print("  BRSABV Ekana Cricket Stadium, Lucknow | April 12, 2026 | 3:30 PM IST")
print("=" * 70)

print("\n--- TEAM PROFILES ---")
print(f"  {'':30s} {'LSG':>12s} {'GT':>12s}")
print(f"  {'Elo Rating':30s} {lsg_elo:>12.1f} {gt_elo:>12.1f}")
print(f"  {'Last 5 Matches (W/5)':30s} {lsg_wins_last5:>12d} {gt_wins_last5:>12d}")
print(f"  {'Momentum (last 5 win%)':30s} {lsg_momentum:>12.1%} {gt_momentum:>12.1%}")
print(f"  {'H2H Record (all-time)':30s} {lsg_h2h_wins:>7d}W/{total_h2h:d}  {gt_h2h_wins:>7d}W/{total_h2h:d}")
print(f"  {'2025 Season':30s} {'6W-8L (7th)':>12s} {'9W-5L (3rd)':>12s}")
print(f"  {'2026 Record':30s} {'2W-1L (5th)':>12s} {'1W-2L (8th)':>12s}")
print(f"  {'Home Advantage':30s} {'YES':>12s} {'NO':>12s}")
print(f"  {'Captain':30s} {'R Pant':>12s} {'S Gill':>12s}")

print("\n--- KEY PLAYERS ---")
print(f"  LSG: Rishabh Pant (c, wk), Nicholas Pooran (explosive power)")
print(f"       Mitchell Marsh (all-rounder), Aiden Markram (top-order)")
print(f"       Abdul Samad, Mukul Choudhary (match-winner, 54 off 27 vs KKR)")
print(f"       Mohammed Shami (pace), Avesh Khan (pace)")
print(f"       Ravi Bishnoi (leg-spin), Manimaran Siddharth (left-arm spin)")
print(f"       Mayank Yadav (express pace), Digvesh Rathi (medium pace)")
print(f"  GT:  Shubman Gill (c, 70 off 45 vs DC), Jos Buttler (116 runs in 3 inn)")
print(f"       Sai Sudharsan (Orange Cap 2025, 759 runs), Rahul Tewatia")
print(f"       Shahrukh Khan, Glenn Phillips (power hitter)")
print(f"       Rashid Khan (9 wkts in LSG-GT H2H), Kagiso Rabada (pace)")
print(f"       Prasidh Krishna (Purple Cap 2025, 25 wkts), Mohammed Siraj")
print(f"       Washington Sundar (off-spin), Jason Holder (all-rounder)")

print("\n--- MODEL PREDICTIONS (P(LSG wins)) ---")
print(f"  Random Forest:         {rf_prob[1]:>6.1%}")
if HAS_XGB:
    print(f"  XGBoost:               {xgb_prob[1]:>6.1%}")
print(f"  Gradient Boosting:     {gb_prob[1]:>6.1%}")
print(f"  Logistic Regression:   {lr_prob[1]:>6.1%}")
print(f"  Ensemble (weighted):   {base_lsg_prob:>6.1%}")

print("\n--- CONTEXTUAL ADJUSTMENTS ---")
for name, adj in adjustments.items():
    if adj != 0:
        direction = "+" if adj > 0 else ""
        print(f"  {name:50s} {direction}{adj:.1%}")
print(f"  {'TOTAL ADJUSTMENT':50s} {'+' if total_adjustment > 0 else ''}{total_adjustment:.1%}")

print(f"\n--- FINAL PREDICTION ---")
print(f"  LSG Win Probability:   {adjusted_lsg_prob:>6.1%}")
print(f"  GT  Win Probability:   {adjusted_gt_prob:>6.1%}")

print(f"\n--- MONTE CARLO SIMULATION ({N_SIM:,d} matches) ---")
print(f"  LSG wins: {lsg_wins_sim:>5,d} ({sim_lsg_pct:.1f}%)")
print(f"  GT  wins: {gt_wins_sim:>5,d} ({sim_gt_pct:.1f}%)")
if lsg_margins:
    print(f"  Avg LSG win margin: {np.mean(lsg_margins):.0f} runs")
if gt_margins:
    print(f"  Avg GT  win margin: {np.mean(gt_margins):.0f} runs")

print("\n" + "=" * 70)
winner = "LSG" if adjusted_lsg_prob > 0.5 else "GT"
winner_full = LSG if winner == "LSG" else GT
loser = "GT" if winner == "LSG" else "LSG"
win_prob = max(adjusted_lsg_prob, adjusted_gt_prob)
confidence_level = "HIGH" if win_prob > 0.62 else "MODERATE" if win_prob > 0.55 else "LEAN"

print(f"  PREDICTED WINNER: {winner} ({winner_full})")
print(f"  WIN PROBABILITY:  {win_prob:.1%}  [{confidence_level}]")
if winner == "LSG" and lsg_margins:
    avg_margin = int(np.mean(lsg_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")
elif winner == "GT" and gt_margins:
    avg_margin = int(np.mean(gt_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")

print("\n--- NARRATIVE ---")
if winner == "LSG":
    print(f"  Lucknow Super Giants leverage home advantage at Ekana to continue")
    print(f"  their impressive 2026 run. Having won the last 3 H2H encounters vs")
    print(f"  GT, LSG hold the psychological edge. Rishabh Pant's aggressive")
    print(f"  captaincy and Mukul Choudhary's match-winning ability are key.")
    print(f"  Mohammed Shami's pace on the Ekana surface troubles GT's top order.")
    print(f"  Despite GT's batting firepower (Gill, Buttler, Sudharsan), LSG's")
    print(f"  bowling depth and home crowd support tip the balance.")
else:
    print(f"  Gujarat Titans' batting firepower proves too much for LSG at Ekana.")
    print(f"  Shubman Gill (back in form with 70 vs DC) and Jos Buttler (116 runs")
    print(f"  in 3 innings) anchor the innings, while Rashid Khan's 9 wickets in")
    print(f"  LSG-GT H2H history gives him a psychological edge. GT's superior")
    print(f"  2025 season (3rd place vs LSG's 7th) reflects deeper squad quality.")
    print(f"  Prasidh Krishna's pace and Rashid's spin combine to restrict LSG.")

print("=" * 70)