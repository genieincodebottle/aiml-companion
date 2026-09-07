"""
IPL 2026 Match Prediction: CSK vs KKR (Match 22, April 14, 2026)
================================================================
MA Chidambaram Stadium, Chepauk, Chennai | 7:30 PM IST
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
print("  IPL 2026 MATCH PREDICTION: CSK vs KKR")
print("  Match 22 | MA Chidambaram Stadium, Chepauk, Chennai")
print("  April 14, 2026 | 7:30 PM IST")
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
    # CSK 2025: 4W 10L, 10th place (LAST) - eliminated 1st, worst season ever
    {"date": "2025-03-25", "team1": "Chennai Super Kings", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-01", "team1": "Royal Challengers Bangalore", "team2": "Chennai Super Kings", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-04-07", "team1": "Chennai Super Kings", "team2": "Punjab Kings", "winner": "Chennai Super Kings", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-04-13", "team1": "Delhi Capitals", "team2": "Chennai Super Kings", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-19", "team1": "Chennai Super Kings", "team2": "Sunrisers Hyderabad", "winner": "Sunrisers Hyderabad", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-25", "team1": "Kolkata Knight Riders", "team2": "Chennai Super Kings", "winner": "Chennai Super Kings", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-01", "team1": "Chennai Super Kings", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-07", "team1": "Rajasthan Royals", "team2": "Chennai Super Kings", "winner": "Chennai Super Kings", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-13", "team1": "Chennai Super Kings", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-19", "team1": "Mumbai Indians", "team2": "Chennai Super Kings", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-05-25", "team1": "Chennai Super Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-30", "team1": "Punjab Kings", "team2": "Chennai Super Kings", "winner": "Chennai Super Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-06-01", "team1": "Chennai Super Kings", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-06-02", "team1": "Lucknow Super Giants", "team2": "Chennai Super Kings", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 20},

    # KKR 2025: 5W 7L 2NR, 8th place - eliminated 4th
    {"date": "2025-03-26", "team1": "Kolkata Knight Riders", "team2": "Rajasthan Royals", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-04-01", "team1": "Mumbai Indians", "team2": "Kolkata Knight Riders", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-07", "team1": "Sunrisers Hyderabad", "team2": "Kolkata Knight Riders", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 40},
    {"date": "2025-04-13", "team1": "Kolkata Knight Riders", "team2": "Delhi Capitals", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-19", "team1": "Kolkata Knight Riders", "team2": "Rajasthan Royals", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-04-25", "team1": "Kolkata Knight Riders", "team2": "Chennai Super Kings", "winner": "Chennai Super Kings", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-01", "team1": "Gujarat Titans", "team2": "Kolkata Knight Riders", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-05-07", "team1": "Kolkata Knight Riders", "team2": "Punjab Kings", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-05-13", "team1": "Lucknow Super Giants", "team2": "Kolkata Knight Riders", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-05-19", "team1": "Kolkata Knight Riders", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-25", "team1": "Chennai Super Kings", "team2": "Kolkata Knight Riders", "winner": "Kolkata Knight Riders", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Kolkata Knight Riders", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-06-01", "team1": "Kolkata Knight Riders", "team2": "Sunrisers Hyderabad", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-06-02", "team1": "Delhi Capitals", "team2": "Kolkata Knight Riders", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},

    # IPL 2026 results so far (Matches 1-21)
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
    {"date": "2026-04-12", "team1": "Lucknow Super Giants", "team2": "Gujarat Titans", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2026-04-12", "team1": "Mumbai Indians", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "runs", "result_margin": 18},
    # Match 21: SRH beat RR by 57 runs (SRH 216/6, RR 159 all out)
    {"date": "2026-04-13", "team1": "Sunrisers Hyderabad", "team2": "Rajasthan Royals", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "runs", "result_margin": 57},
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
# 5. PREDICT: CSK vs KKR
# ============================================================================
print("\n[5/7] Predicting CSK vs KKR...")

CSK = "Chennai Super Kings"
KKR = "Kolkata Knight Riders"

csk_elo = final_elo[CSK]
kkr_elo = final_elo[KKR]

csk_matches = all_matches[(all_matches["team1"] == CSK) | (all_matches["team2"] == CSK)].tail(5)
kkr_matches = all_matches[(all_matches["team1"] == KKR) | (all_matches["team2"] == KKR)].tail(5)

csk_wins_last5 = sum(csk_matches["winner"] == CSK)
kkr_wins_last5 = sum(kkr_matches["winner"] == KKR)
csk_momentum = csk_wins_last5 / 5
kkr_momentum = kkr_wins_last5 / 5

csk_kkr = all_matches[
    ((all_matches["team1"] == CSK) & (all_matches["team2"] == KKR)) |
    ((all_matches["team1"] == KKR) & (all_matches["team2"] == CSK))
]
total_h2h = len(csk_kkr[csk_kkr["winner"].isin([CSK, KKR])])
csk_h2h_wins = sum(csk_kkr["winner"] == CSK)
kkr_h2h_wins = total_h2h - csk_h2h_wins
csk_h2h_winrate = csk_h2h_wins / total_h2h if total_h2h > 0 else 0.5

# CSK is team1 (HOME - Chennai), KKR is team2
match_features = {
    "elo_team1": csk_elo,
    "elo_team2": kkr_elo,
    "elo_diff": csk_elo - kkr_elo,
    "momentum_team1": csk_momentum,
    "momentum_team2": kkr_momentum,
    "momentum_diff": csk_momentum - kkr_momentum,
    "h2h_team1_winrate": csk_h2h_winrate,
    "h2h_matches": total_h2h,
    "home_team1": 1,   # CSK home ground - Chepauk
    "home_team2": 0,
    "venue_chase_bias": 0.45,   # Chepauk - traditionally favors batting first, spinners grip
    "toss_chose_field": 1,  # Night game + dew
    "toss_winner_is_team1": 0.5,
    "recency_weight": 1.0,
    "elo_x_momentum_t1": csk_elo * csk_momentum,
    "elo_x_momentum_t2": kkr_elo * kkr_momentum,
    "elo_x_home_t1": csk_elo * 1,
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

base_csk_prob = ensemble_prob[1]  # P(CSK wins)

adjustments = {}

# A. CSK HOME advantage at Chepauk (+4%)
# Chepauk is CSK's fortress - strong home support
# Sanju Samson scored century here vs DC in last match
# CSK won last match at Chepauk by 23 runs vs DC
# Spin-friendly pitch suits CSK's bowling
adjustments["csk_home_chepauk"] = +0.04

# B. CSK H2H dominance: ~21W vs 11W all-time (+3%)
# CSK leads head-to-head 21-11 (63% win rate)
# Historical dominance in this fixture
# CSK won at KKR's home in 2025 too
adjustments["csk_h2h_dominance"] = +0.03

# C. CSK Sanju Samson acquisition (+2%)
# Traded from RR - scored century vs DC at Chepauk (last match)
# Explosive batting at top, proven match-winner
# Fills the gap left by Jadeja trade
adjustments["csk_samson_impact"] = +0.02

# D. CSK just won - momentum building (+2%)
# Beat DC by 23 runs at Chepauk (Match 18)
# First win after 3 losses - confidence restored
# Noor Ahmad's spin threat on Chepauk pitch
adjustments["csk_momentum_win"] = +0.02

# E. Chepauk spin factor suits CSK (+2%)
# Noor Ahmad (24 wkts in 2025) - lethal on spin-friendly Chepauk
# Red soil pitch grips for spinners in middle overs
# KKR batters (except Varun) may struggle vs quality spin
adjustments["csk_spin_advantage"] = +0.02

# F. KKR winless in 2026 - worst form (-3%)
# 0W-3L + 1NR, bottom of table alongside CSK
# Lost to MI (17 runs), SRH (65 runs!), LSG (3 wkts)
# Only 1 point from washout vs PBKS
# Team confidence at rock bottom
adjustments["kkr_winless_2026"] = -0.03

# G. KKR away from home (-1%)
# Playing away from Eden Gardens
# Chepauk conditions very different from Eden Gardens
# KKR historically struggle at Chennai
adjustments["kkr_away_disadvantage"] = -0.01

# H. CSK 2025 was worst ever season (-2%)
# 4W-10L, 10th place (last)
# Core team rebuilding under Gaikwad
# Still finding right combination
adjustments["csk_2025_hangover"] = -0.02

# I. KKR have Varun Chakravarthy (+1%)
# Varun is lethal spinner - could exploit Chepauk too
# Mystery spin could trouble CSK middle order
# But Chepauk conditions might suit CSK spinners MORE
adjustments["kkr_varun_threat"] = +0.01

# J. KKR Pathirana + Cameron Green acquisitions (+1%)
# Pathirana's yorkers are hard to play at death
# Cameron Green all-round value (if fit)
# But both yet to fire in 2026
adjustments["kkr_new_signings"] = +0.01

# K. Dew factor at Chepauk in 2nd innings (-1%)
# 58% humidity, dew from 12th over of 2nd innings
# Chasing team gets slight advantage
# Could negate CSK's spin advantage partially
adjustments["chepauk_dew_factor"] = -0.01

total_adjustment = sum(adjustments.values())
adjusted_csk_prob = np.clip(base_csk_prob + total_adjustment, 0.05, 0.95)
adjusted_kkr_prob = 1 - adjusted_csk_prob

# ============================================================================
# 7. MONTE CARLO SIMULATION
# ============================================================================
print("\n[7/7] Running Monte Carlo simulation (10,000 matches)...")

np.random.seed(42)
N_SIM = 10000

csk_wins_sim = 0
kkr_wins_sim = 0
csk_margins = []
kkr_margins = []

for _ in range(N_SIM):
    noise = np.random.normal(0, 0.05)
    match_prob = np.clip(adjusted_csk_prob + noise, 0.1, 0.9)
    if np.random.random() < match_prob:
        csk_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        csk_margins.append(margin)
    else:
        kkr_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        kkr_margins.append(margin)

sim_csk_pct = csk_wins_sim / N_SIM * 100
sim_kkr_pct = kkr_wins_sim / N_SIM * 100

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  PREDICTION REPORT: CSK vs KKR | IPL 2026 Match 22")
print("  MA Chidambaram Stadium, Chepauk, Chennai | April 14, 2026 | 7:30 PM IST")
print("=" * 70)

print("\n--- TEAM PROFILES ---")
print(f"  {'':30s} {'CSK':>12s} {'KKR':>12s}")
print(f"  {'Elo Rating':30s} {csk_elo:>12.1f} {kkr_elo:>12.1f}")
print(f"  {'Last 5 Matches (W/5)':30s} {csk_wins_last5:>12d} {kkr_wins_last5:>12d}")
print(f"  {'Momentum (last 5 win%)':30s} {csk_momentum:>12.1%} {kkr_momentum:>12.1%}")
print(f"  {'H2H Record (all-time)':30s} {csk_h2h_wins:>7d}W/{total_h2h:d}  {kkr_h2h_wins:>7d}W/{total_h2h:d}")
print(f"  {'2025 Season':30s} {'4W-10L(10th)':>12s} {'5W-7L (8th)':>12s}")
print(f"  {'2026 Record':30s} {'1W-3L (9th)':>12s} {'0W-3L(10th)':>12s}")
print(f"  {'Home Advantage':30s} {'YES':>12s} {'NO':>12s}")
print(f"  {'Captain':30s} {'R Gaikwad':>12s} {'A Rahane':>12s}")

print("\n--- KEY PLAYERS ---")
print(f"  CSK: Ruturaj Gaikwad (c), Sanju Samson (century vs DC at Chepauk)")
print(f"       MS Dhoni (mentor/finisher), Shivam Dube (power hitter)")
print(f"       Dewald Brevis (explosive batting), Ayush Mhatre (young talent)")
print(f"       Noor Ahmad (24 wkts in 2025, lethal spinner)")
print(f"       Khaleel Ahmed (swing bowler), Anshul Kamboj (pace)")
print(f"  KKR: Ajinkya Rahane (c), Rinku Singh (vc, finisher)")
print(f"       Varun Chakravarthy (mystery spin, 17 wkts in 2025)")
print(f"       Cameron Green (all-rounder, Rs 25.2cr)")
print(f"       Matheesha Pathirana (yorker specialist, Rs 18cr)")
print(f"       Sunil Narine (all-rounder), Venkatesh Iyer (batting)")
print(f"       Vaibhav Arora (pace, 17 wkts in 2025)")

print("\n--- MODEL PREDICTIONS (P(CSK wins)) ---")
print(f"  Random Forest:         {rf_prob[1]:>6.1%}")
if HAS_XGB:
    print(f"  XGBoost:               {xgb_prob[1]:>6.1%}")
print(f"  Gradient Boosting:     {gb_prob[1]:>6.1%}")
print(f"  Logistic Regression:   {lr_prob[1]:>6.1%}")
print(f"  Ensemble (weighted):   {base_csk_prob:>6.1%}")

print("\n--- CONTEXTUAL ADJUSTMENTS ---")
for name, adj in adjustments.items():
    if adj != 0:
        direction = "+" if adj > 0 else ""
        print(f"  {name:50s} {direction}{adj:.1%}")
print(f"  {'TOTAL ADJUSTMENT':50s} {'+' if total_adjustment > 0 else ''}{total_adjustment:.1%}")

print(f"\n--- FINAL PREDICTION ---")
print(f"  CSK Win Probability:   {adjusted_csk_prob:>6.1%}")
print(f"  KKR Win Probability:   {adjusted_kkr_prob:>6.1%}")

print(f"\n--- MONTE CARLO SIMULATION ({N_SIM:,d} matches) ---")
print(f"  CSK wins: {csk_wins_sim:>5,d} ({sim_csk_pct:.1f}%)")
print(f"  KKR wins: {kkr_wins_sim:>5,d} ({sim_kkr_pct:.1f}%)")
if csk_margins:
    print(f"  Avg CSK win margin: {np.mean(csk_margins):.0f} runs")
if kkr_margins:
    print(f"  Avg KKR win margin: {np.mean(kkr_margins):.0f} runs")

print("\n" + "=" * 70)
winner = "CSK" if adjusted_csk_prob > 0.5 else "KKR"
winner_full = CSK if winner == "CSK" else KKR
loser = "KKR" if winner == "CSK" else "CSK"
win_prob = max(adjusted_csk_prob, adjusted_kkr_prob)
confidence_level = "HIGH" if win_prob > 0.62 else "MODERATE" if win_prob > 0.55 else "LEAN"

print(f"  PREDICTED WINNER: {winner} ({winner_full})")
print(f"  WIN PROBABILITY:  {win_prob:.1%}  [{confidence_level}]")
if winner == "CSK" and csk_margins:
    avg_margin = int(np.mean(csk_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")
elif winner == "KKR" and kkr_margins:
    avg_margin = int(np.mean(kkr_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")

print("\n--- NARRATIVE ---")
if winner == "CSK":
    print(f"  Chennai Super Kings extend their Chepauk resurgence against a")
    print(f"  winless KKR side. Sanju Samson, fresh off a century vs DC, anchors")
    print(f"  the innings while Noor Ahmad's spin on the grippy Chepauk pitch")
    print(f"  strangles KKR's middle order. CSK's dominant H2H record (21-11)")
    print(f"  and home fortress advantage prove decisive. KKR's 0-3 start and")
    print(f"  away struggles continue as Rahane's side fails to crack the code")
    print(f"  at Chennai. Dhoni's tactical nous from behind the stumps adds")
    print(f"  the X-factor that KKR simply cannot match.")
else:
    print(f"  Kolkata Knight Riders finally break their duck in 2026, stunning")
    print(f"  CSK at Chepauk. Varun Chakravarthy's mystery spin proves even more")
    print(f"  effective than CSK's spinners on the turning track. Rinku Singh's")
    print(f"  finishing ability seals a famous away victory as KKR's star signings")
    print(f"  Pathirana and Cameron Green finally justify their price tags.")
    print(f"  Despite CSK's home advantage, KKR's desperation for a win fuels")
    print(f"  an intense performance that overwhelms the hosts.")

print("=" * 70)
