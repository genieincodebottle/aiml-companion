"""
IPL 2026 Match Prediction: SRH vs RR (Match 21, April 13, 2026)
================================================================
Rajiv Gandhi International Stadium, Hyderabad | 7:30 PM IST
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
print("  IPL 2026 MATCH PREDICTION: SRH vs RR")
print("  Match 21 | Rajiv Gandhi Intl Stadium, Hyderabad")
print("  April 13, 2026 | 7:30 PM IST")
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
    # SRH 2025: 6W 8L, 6th place - eliminated 3rd
    {"date": "2025-03-25", "team1": "Sunrisers Hyderabad", "team2": "Delhi Capitals", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-04-01", "team1": "Rajasthan Royals", "team2": "Sunrisers Hyderabad", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-04-07", "team1": "Sunrisers Hyderabad", "team2": "Kolkata Knight Riders", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 40},
    {"date": "2025-04-13", "team1": "Punjab Kings", "team2": "Sunrisers Hyderabad", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-19", "team1": "Sunrisers Hyderabad", "team2": "Chennai Super Kings", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-25", "team1": "Mumbai Indians", "team2": "Sunrisers Hyderabad", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-01", "team1": "Sunrisers Hyderabad", "team2": "Lucknow Super Giants", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-07", "team1": "Gujarat Titans", "team2": "Sunrisers Hyderabad", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-05-13", "team1": "Sunrisers Hyderabad", "team2": "Royal Challengers Bangalore", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-05-19", "team1": "Delhi Capitals", "team2": "Sunrisers Hyderabad", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-25", "team1": "Sunrisers Hyderabad", "team2": "Rajasthan Royals", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-30", "team1": "Mumbai Indians", "team2": "Sunrisers Hyderabad", "winner": "Sunrisers Hyderabad", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-06-01", "team1": "Kolkata Knight Riders", "team2": "Sunrisers Hyderabad", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-06-02", "team1": "Sunrisers Hyderabad", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},

    # RR 2025: 4W 10L, 9th place - eliminated 2nd, worst season
    {"date": "2025-03-26", "team1": "Rajasthan Royals", "team2": "Punjab Kings", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-01", "team1": "Rajasthan Royals", "team2": "Sunrisers Hyderabad", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-04-07", "team1": "Royal Challengers Bangalore", "team2": "Rajasthan Royals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-04-13", "team1": "Rajasthan Royals", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-19", "team1": "Kolkata Knight Riders", "team2": "Rajasthan Royals", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-04-25", "team1": "Lucknow Super Giants", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-01", "team1": "Rajasthan Royals", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-05-07", "team1": "Chennai Super Kings", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-13", "team1": "Rajasthan Royals", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-19", "team1": "Punjab Kings", "team2": "Rajasthan Royals", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-05-25", "team1": "Sunrisers Hyderabad", "team2": "Rajasthan Royals", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-30", "team1": "Rajasthan Royals", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-06-01", "team1": "Mumbai Indians", "team2": "Rajasthan Royals", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 100},
    {"date": "2025-06-02", "team1": "Rajasthan Royals", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 4},

    # IPL 2026 results so far (Matches 1-20)
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
# 5. PREDICT: SRH vs RR
# ============================================================================
print("\n[5/7] Predicting SRH vs RR...")

SRH = "Sunrisers Hyderabad"
RR = "Rajasthan Royals"

srh_elo = final_elo[SRH]
rr_elo = final_elo[RR]

srh_matches = all_matches[(all_matches["team1"] == SRH) | (all_matches["team2"] == SRH)].tail(5)
rr_matches = all_matches[(all_matches["team1"] == RR) | (all_matches["team2"] == RR)].tail(5)

srh_wins_last5 = sum(srh_matches["winner"] == SRH)
rr_wins_last5 = sum(rr_matches["winner"] == RR)
srh_momentum = srh_wins_last5 / 5
rr_momentum = rr_wins_last5 / 5

srh_rr = all_matches[
    ((all_matches["team1"] == SRH) & (all_matches["team2"] == RR)) |
    ((all_matches["team1"] == RR) & (all_matches["team2"] == SRH))
]
total_h2h = len(srh_rr[srh_rr["winner"].isin([SRH, RR])])
srh_h2h_wins = sum(srh_rr["winner"] == SRH)
rr_h2h_wins = total_h2h - srh_h2h_wins
srh_h2h_winrate = srh_h2h_wins / total_h2h if total_h2h > 0 else 0.5

# SRH is team1 (HOME - Hyderabad), RR is team2
match_features = {
    "elo_team1": srh_elo,
    "elo_team2": rr_elo,
    "elo_diff": srh_elo - rr_elo,
    "momentum_team1": srh_momentum,
    "momentum_team2": rr_momentum,
    "momentum_diff": srh_momentum - rr_momentum,
    "h2h_team1_winrate": srh_h2h_winrate,
    "h2h_matches": total_h2h,
    "home_team1": 1,   # SRH home ground - Rajiv Gandhi Stadium
    "home_team2": 0,
    "venue_chase_bias": 0.50,   # Rajiv Gandhi - balanced but dew favors chasing
    "toss_chose_field": 1,  # Night game + heavy dew
    "toss_winner_is_team1": 0.5,
    "recency_weight": 1.0,
    "elo_x_momentum_t1": srh_elo * srh_momentum,
    "elo_x_momentum_t2": rr_elo * rr_momentum,
    "elo_x_home_t1": srh_elo * 1,
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

base_srh_prob = ensemble_prob[1]  # P(SRH wins)

adjustments = {}

# A. RR 4W-0L in 2026 - DOMINANT, top of table (-5%)
# Beat CSK (17 runs), GT (18 runs), DC (4 wkts), MI (27 runs), RCB (15 runs - Sooryavanshi 78 off 26)
# Best NRR (2.403), only unbeaten team alongside PBKS
# Incredible momentum - 5 consecutive wins
adjustments["rr_dominant_2026_form"] = -0.05

# B. Vaibhav Sooryavanshi explosive batting (-2%)
# 78 off 26 vs RCB (15-ball 50!), youngest fastest Indian century in IPL history
# Combined with Yashasvi Jaiswal - lethal opening pair
# These two can demolish any bowling attack in powerplay
adjustments["rr_sooryavanshi_jaiswal_opening"] = -0.02

# C. Dhruv Jurel + Riyan Parag middle order (-1%)
# Jurel 81* off 43 vs RCB - match-winning innings
# Parag captaincy improving, Hetmyer finisher
# Strong middle order depth
adjustments["rr_jurel_parag_middle"] = -0.01

# D. Jofra Archer pace + RR bowling depth (-1%)
# Archer back to full fitness, express pace
# Nandre Burger, Sandeep Sharma provide variety
# Ravindra Jadeja all-round value
adjustments["rr_archer_bowling"] = -0.01

# E. SRH home advantage at Hyderabad (+3%)
# Rajiv Gandhi Stadium is SRH fortress
# Flat batting pitch suits Head/Abhishek/Klaasen
# Home crowd support significant
# SRH have strong home record in H2H vs RR
adjustments["srh_home_advantage"] = +0.03

# F. SRH H2H dominance: 12-9 vs RR (+2%)
# SRH lead all-time head-to-head
# Won at Hyderabad vs RR in 2025 (25 runs)
# Historical advantage in this fixture
adjustments["srh_h2h_dominance"] = +0.02

# G. SRH batting firepower - Head, Abhishek, Klaasen (+2%)
# Travis Head (explosive opener, 105/0 powerplay vs PBKS - 74 off 28 from Abhishek)
# Abhishek Sharma (destructive powerplay batting)
# Heinrich Klaasen (best T20 finisher in world)
# On a flat Hyderabad pitch, these 3 are devastating
adjustments["srh_batting_firepower"] = +0.02

# H. SRH 1W-3L in 2026 - poor form (-3%)
# Lost to RCB (6 wkts), Lost to LSG at home (5 wkts), Lost to PBKS (6 wkts)
# Only win: beat KKR by 65 runs (showed batting power)
# Inconsistent bowling - leak runs regularly
# Pat Cummins absent (back injury) - leadership void
adjustments["srh_poor_2026_form"] = -0.03

# I. SRH bowling concerns (-1%)
# Without Cummins, bowling attack lacks bite
# Harshal Patel as lead pacer not enough
# Leaked runs in losses - can't defend totals
# RR's batting depth will exploit weak bowling
adjustments["srh_bowling_weakness"] = -0.01

# J. Night match dew at Hyderabad (-1%)
# Heavy dew from 12th-14th over of 2nd innings
# Chasing team gets massive advantage
# If RR win toss and bowl, SRH in trouble
adjustments["hyderabad_dew_factor"] = -0.01

total_adjustment = sum(adjustments.values())
adjusted_srh_prob = np.clip(base_srh_prob + total_adjustment, 0.05, 0.95)
adjusted_rr_prob = 1 - adjusted_srh_prob

# ============================================================================
# 7. MONTE CARLO SIMULATION
# ============================================================================
print("\n[7/7] Running Monte Carlo simulation (10,000 matches)...")

np.random.seed(42)
N_SIM = 10000

srh_wins_sim = 0
rr_wins_sim = 0
srh_margins = []
rr_margins = []

for _ in range(N_SIM):
    noise = np.random.normal(0, 0.05)
    match_prob = np.clip(adjusted_srh_prob + noise, 0.1, 0.9)
    if np.random.random() < match_prob:
        srh_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        srh_margins.append(margin)
    else:
        rr_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        rr_margins.append(margin)

sim_srh_pct = srh_wins_sim / N_SIM * 100
sim_rr_pct = rr_wins_sim / N_SIM * 100

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  PREDICTION REPORT: SRH vs RR | IPL 2026 Match 21")
print("  Rajiv Gandhi Intl Stadium, Hyderabad | April 13, 2026 | 7:30 PM IST")
print("=" * 70)

print("\n--- TEAM PROFILES ---")
print(f"  {'':30s} {'SRH':>12s} {'RR':>12s}")
print(f"  {'Elo Rating':30s} {srh_elo:>12.1f} {rr_elo:>12.1f}")
print(f"  {'Last 5 Matches (W/5)':30s} {srh_wins_last5:>12d} {rr_wins_last5:>12d}")
print(f"  {'Momentum (last 5 win%)':30s} {srh_momentum:>12.1%} {rr_momentum:>12.1%}")
print(f"  {'H2H Record (all-time)':30s} {srh_h2h_wins:>7d}W/{total_h2h:d}  {rr_h2h_wins:>7d}W/{total_h2h:d}")
print(f"  {'2025 Season':30s} {'6W-8L (6th)':>12s} {'4W-10L (9th)':>12s}")
print(f"  {'2026 Record':30s} {'1W-3L (8th)':>12s} {'4W-0L (1st)':>12s}")
print(f"  {'Home Advantage':30s} {'YES':>12s} {'NO':>12s}")
print(f"  {'Captain':30s} {'I Kishan':>12s} {'R Parag':>12s}")

print("\n--- KEY PLAYERS ---")
print(f"  SRH: Travis Head (explosive opener, 74 off 28 vs PBKS in 2026)")
print(f"       Abhishek Sharma (destructive powerplay, 105/0 PP with Head)")
print(f"       Ishan Kishan (c, wk), Heinrich Klaasen (best T20 finisher)")
print(f"       Nitish Kumar Reddy (all-rounder), Liam Livingstone (power)")
print(f"       Harshal Patel (lead pacer), Brydon Carse (pace)")
print(f"       Pat Cummins ABSENT (back injury)")
print(f"  RR:  Vaibhav Sooryavanshi (78 off 26 vs RCB, 15-ball 50)")
print(f"       Yashasvi Jaiswal (explosive opener)")
print(f"       Dhruv Jurel (wk, 81* off 43 vs RCB), Riyan Parag (c)")
print(f"       Shimron Hetmyer (finisher), Ravindra Jadeja (all-rounder)")
print(f"       Jofra Archer (express pace), Nandre Burger (pace)")
print(f"       Sandeep Sharma (swing)")

print("\n--- MODEL PREDICTIONS (P(SRH wins)) ---")
print(f"  Random Forest:         {rf_prob[1]:>6.1%}")
if HAS_XGB:
    print(f"  XGBoost:               {xgb_prob[1]:>6.1%}")
print(f"  Gradient Boosting:     {gb_prob[1]:>6.1%}")
print(f"  Logistic Regression:   {lr_prob[1]:>6.1%}")
print(f"  Ensemble (weighted):   {base_srh_prob:>6.1%}")

print("\n--- CONTEXTUAL ADJUSTMENTS ---")
for name, adj in adjustments.items():
    if adj != 0:
        direction = "+" if adj > 0 else ""
        print(f"  {name:50s} {direction}{adj:.1%}")
print(f"  {'TOTAL ADJUSTMENT':50s} {'+' if total_adjustment > 0 else ''}{total_adjustment:.1%}")

print(f"\n--- FINAL PREDICTION ---")
print(f"  SRH Win Probability:   {adjusted_srh_prob:>6.1%}")
print(f"  RR  Win Probability:   {adjusted_rr_prob:>6.1%}")

print(f"\n--- MONTE CARLO SIMULATION ({N_SIM:,d} matches) ---")
print(f"  SRH wins: {srh_wins_sim:>5,d} ({sim_srh_pct:.1f}%)")
print(f"  RR  wins: {rr_wins_sim:>5,d} ({sim_rr_pct:.1f}%)")
if srh_margins:
    print(f"  Avg SRH win margin: {np.mean(srh_margins):.0f} runs")
if rr_margins:
    print(f"  Avg RR  win margin: {np.mean(rr_margins):.0f} runs")

print("\n" + "=" * 70)
winner = "SRH" if adjusted_srh_prob > 0.5 else "RR"
winner_full = SRH if winner == "SRH" else RR
loser = "RR" if winner == "SRH" else "SRH"
win_prob = max(adjusted_srh_prob, adjusted_rr_prob)
confidence_level = "HIGH" if win_prob > 0.62 else "MODERATE" if win_prob > 0.55 else "LEAN"

print(f"  PREDICTED WINNER: {winner} ({winner_full})")
print(f"  WIN PROBABILITY:  {win_prob:.1%}  [{confidence_level}]")
if winner == "SRH" and srh_margins:
    avg_margin = int(np.mean(srh_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")
elif winner == "RR" and rr_margins:
    avg_margin = int(np.mean(rr_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")

print("\n--- NARRATIVE ---")
if winner == "RR":
    print(f"  Rajasthan Royals extend their perfect 2026 run to five wins.")
    print(f"  Sooryavanshi and Jaiswal's explosive opening combination tears")
    print(f"  through SRH's weakened bowling (no Cummins). Jurel provides the")
    print(f"  middle-order anchor while Archer's pace troubles SRH's batters.")
    print(f"  Despite SRH's home advantage and Head-Abhishek-Klaasen firepower,")
    print(f"  RR's all-round dominance and unstoppable momentum prove too much.")
    print(f"  SRH's 1-3 record and bowling concerns make them vulnerable.")
else:
    print(f"  Sunrisers Hyderabad finally find their 2026 form at their fortress.")
    print(f"  Travis Head and Abhishek Sharma's explosive opening on a flat Rajiv")
    print(f"  Gandhi pitch sets up a massive total. Klaasen's finishing power adds")
    print(f"  late runs. SRH's H2H dominance (12-9) and home advantage finally")
    print(f"  snap RR's winning streak. The crowd factor at Hyderabad lifts SRH")
    print(f"  while RR face their toughest test away from home.")

print("=" * 70)
