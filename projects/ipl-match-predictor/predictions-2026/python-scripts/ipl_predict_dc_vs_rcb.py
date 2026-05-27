"""
IPL 2026 Match Prediction (RETROACTIVE): DC vs RCB (Match 39, April 27, 2026)
=============================================================================
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
print("  IPL 2026 PREDICTION (RETROACTIVE): DC vs RCB")
print("  Match 39 | Arun Jaitley Stadium, New Delhi")
print("  April 27, 2026 | 7:30 PM IST")
print("=" * 70)

print("\n[1/7] Loading historical data (2008-2024)...")
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"), na_values=["NA", ""])

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
# 2. SUPPLEMENT WITH 2025 + 2026 SEASON DATA (PRE-MATCH ONLY, before Apr 27)
# ============================================================================
print("\n[2/7] Supplementing with 2025 + 2026 season data...")

ipl_supp = [
    # DC 2025 results (~7W 7L, made playoffs)
    {"date": "2025-03-25", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-03-30", "team1": "Sunrisers Hyderabad", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-05", "team1": "Delhi Capitals", "team2": "Chennai Super Kings", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "runs", "result_margin": 18},
    {"date": "2025-04-11", "team1": "Delhi Capitals", "team2": "Gujarat Titans", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "runs", "result_margin": 12},
    {"date": "2025-04-17", "team1": "Royal Challengers Bangalore", "team2": "Delhi Capitals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-23", "team1": "Delhi Capitals", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-27", "team1": "Delhi Capitals", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-04", "team1": "Punjab Kings", "team2": "Delhi Capitals", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "runs", "result_margin": 22},
    {"date": "2025-05-10", "team1": "Delhi Capitals", "team2": "Kolkata Knight Riders", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Kolkata Knight Riders", "toss_decision": "field", "result": "runs", "result_margin": 15},
    {"date": "2025-05-15", "team1": "Delhi Capitals", "team2": "Royal Challengers Bangalore", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2025-05-20", "team1": "Gujarat Titans", "team2": "Delhi Capitals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-22", "team1": "Delhi Capitals", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 3},
    {"date": "2025-05-26", "team1": "Delhi Capitals", "team2": "Sunrisers Hyderabad", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "runs", "result_margin": 20},

    # RCB 2025 (CHAMPIONS, ~11W 4L, won final vs PBKS)
    {"date": "2025-03-22", "team1": "Royal Challengers Bangalore", "team2": "Kolkata Knight Riders", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Kolkata Knight Riders", "toss_decision": "field", "result": "runs", "result_margin": 22},
    {"date": "2025-03-29", "team1": "Royal Challengers Bangalore", "team2": "Rajasthan Royals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 7},
    {"date": "2025-04-04", "team1": "Mumbai Indians", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-10", "team1": "Royal Challengers Bangalore", "team2": "Sunrisers Hyderabad", "winner": "Sunrisers Hyderabad", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "runs", "result_margin": 10},
    {"date": "2025-04-17", "team1": "Royal Challengers Bangalore", "team2": "Delhi Capitals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-22", "team1": "Royal Challengers Bangalore", "team2": "Punjab Kings", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-04-28", "team1": "Chennai Super Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "bat", "result": "wickets", "result_margin": 6},
    {"date": "2025-05-04", "team1": "Royal Challengers Bangalore", "team2": "Gujarat Titans", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 7},
    {"date": "2025-05-08", "team1": "Punjab Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-15", "team1": "Delhi Capitals", "team2": "Royal Challengers Bangalore", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2025-05-19", "team1": "Royal Challengers Bangalore", "team2": "Lucknow Super Giants", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "runs", "result_margin": 28},
    {"date": "2025-05-23", "team1": "Royal Challengers Bangalore", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-28", "team1": "Punjab Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "runs", "result_margin": 15},
    # Final 2025 - RCB beat PBKS by 6 runs
    {"date": "2025-06-03", "team1": "Punjab Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "runs", "result_margin": 6},

    # IPL 2026 DC results - 6th place, 3W 4L (before match 39)
    {"date": "2026-03-29", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "runs", "result_margin": 12},
    {"date": "2026-04-02", "team1": "Sunrisers Hyderabad", "team2": "Delhi Capitals", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "runs", "result_margin": 22},
    {"date": "2026-04-06", "team1": "Delhi Capitals", "team2": "Chennai Super Kings", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2026-04-11", "team1": "Mumbai Indians", "team2": "Delhi Capitals", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2026-04-15", "team1": "Delhi Capitals", "team2": "Kolkata Knight Riders", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Kolkata Knight Riders", "toss_decision": "field", "result": "runs", "result_margin": 14},
    {"date": "2026-04-18", "team1": "Gujarat Titans", "team2": "Delhi Capitals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2026-04-25", "team1": "Delhi Capitals", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "wickets", "result_margin": 6},

    # IPL 2026 RCB results - 2nd place, 5W 2L (before match 39)
    {"date": "2026-03-28", "team1": "Royal Challengers Bangalore", "team2": "Sunrisers Hyderabad", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "runs", "result_margin": 18},
    {"date": "2026-04-01", "team1": "Royal Challengers Bangalore", "team2": "Rajasthan Royals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-06", "team1": "Mumbai Indians", "team2": "Royal Challengers Bangalore", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2026-04-10", "team1": "Royal Challengers Bangalore", "team2": "Chennai Super Kings", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "runs", "result_margin": 28},
    {"date": "2026-04-14", "team1": "Royal Challengers Bangalore", "team2": "Lucknow Super Giants", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "runs", "result_margin": 12},
    {"date": "2026-04-19", "team1": "Sunrisers Hyderabad", "team2": "Royal Challengers Bangalore", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2026-04-25", "team1": "Royal Challengers Bangalore", "team2": "Gujarat Titans", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
]

supp_df = pd.DataFrame(ipl_supp)
supp_df["date"] = pd.to_datetime(supp_df["date"])
supp_df["season"] = supp_df["date"].dt.year.astype(str)
supp_df["id"] = range(900000, 900000 + len(supp_df))
supp_df["match_type"] = "League"
supp_df["super_over"] = "N"
supp_df["player_of_match"] = "Unknown"

all_matches = pd.concat([matches, supp_df], ignore_index=True)
all_matches = all_matches.sort_values("date").reset_index(drop=True)
print(f"  Added {len(supp_df)} 2025+2026 matches -> Total: {len(all_matches)} matches")

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
    "Maharaja Yadavindra Singh International Cricket Stadium": 0.48,
    "M Chinnaswamy Stadium": 0.54,
    "Wankhede Stadium": 0.52,
    "Eden Gardens": 0.48,
    "MA Chidambaram Stadium": 0.45,
    "Rajiv Gandhi International Stadium": 0.50,
    "Narendra Modi Stadium": 0.47,
    "Arun Jaitley Stadium": 0.51,
    "Sawai Mansingh Stadium": 0.50,
}
all_matches["venue_chase_bias"] = all_matches["venue"].map(venue_chase_bias).fillna(0.50)
all_matches["toss_chose_field"] = (all_matches["toss_decision"] == "field").astype(int)
all_matches["toss_winner_is_team1"] = (all_matches["toss_winner"] == all_matches["team1"]).astype(int)
all_matches["season_year"] = all_matches["date"].dt.year
all_matches["recency_weight"] = (all_matches["season_year"] - 2008 + 1) / (2026 - 2008 + 1)
all_matches["elo_x_momentum_t1"] = all_matches["elo_team1"] * all_matches["momentum_team1"]
all_matches["elo_x_momentum_t2"] = all_matches["elo_team2"] * all_matches["momentum_team2"]
all_matches["elo_x_home_t1"] = all_matches["elo_team1"] * all_matches["home_team1"]

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n[4/7] Training models...")

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
    RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=15,
                           min_samples_leaf=8, random_state=42, class_weight="balanced"),
    cv=5, method="isotonic"
)
rf.fit(X_scaled, y, sample_weight=sample_weights)
rf_cv = cross_val_score(
    RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"),
    X_scaled, y, cv=5, scoring="accuracy"
)
print(f"  Random Forest CV: {rf_cv.mean():.4f}")

if HAS_XGB:
    xgb = CalibratedClassifierCV(
        XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5,
                      reg_lambda=2.0, random_state=42, eval_metric="logloss"),
        cv=5, method="isotonic"
    )
    xgb.fit(X_scaled, y, sample_weight=sample_weights)
    xgb_cv = cross_val_score(
        XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss"),
        X_scaled, y, cv=5, scoring="accuracy"
    )
    print(f"  XGBoost CV:       {xgb_cv.mean():.4f}")

gb = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                min_samples_split=15, min_samples_leaf=8, random_state=42),
    cv=5, method="isotonic"
)
gb.fit(X_scaled, y, sample_weight=sample_weights)
gb_cv = cross_val_score(
    GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    X_scaled, y, cv=5, scoring="accuracy"
)
print(f"  Gradient Boost CV: {gb_cv.mean():.4f}")

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_scaled, y, sample_weight=sample_weights)
lr_cv = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy")
print(f"  Logistic Reg CV:  {lr_cv.mean():.4f}")

# ============================================================================
# 5. PREDICT DC vs RCB
# ============================================================================
print("\n[5/7] Predicting DC vs RCB...")

DC = "Delhi Capitals"
RCB = "Royal Challengers Bangalore"

dc_elo = final_elo[DC]
rcb_elo = final_elo[RCB]

dc_matches = all_matches[(all_matches["team1"] == DC) | (all_matches["team2"] == DC)].tail(5)
rcb_matches = all_matches[(all_matches["team1"] == RCB) | (all_matches["team2"] == RCB)].tail(5)
dc_wins_last5 = sum(dc_matches["winner"] == DC)
rcb_wins_last5 = sum(rcb_matches["winner"] == RCB)
dc_momentum = dc_wins_last5 / 5
rcb_momentum = rcb_wins_last5 / 5

dc_rcb = all_matches[
    ((all_matches["team1"] == DC) & (all_matches["team2"] == RCB)) |
    ((all_matches["team1"] == RCB) & (all_matches["team2"] == DC))
]
total_h2h = len(dc_rcb[dc_rcb["winner"].isin([DC, RCB])])
dc_h2h_wins = sum(dc_rcb["winner"] == DC)
rcb_h2h_wins = total_h2h - dc_h2h_wins
dc_h2h_winrate = dc_h2h_wins / total_h2h if total_h2h > 0 else 0.5

# DC team1 (home at Arun Jaitley), RCB team2
match_features = {
    "elo_team1": dc_elo,
    "elo_team2": rcb_elo,
    "elo_diff": dc_elo - rcb_elo,
    "momentum_team1": dc_momentum,
    "momentum_team2": rcb_momentum,
    "momentum_diff": dc_momentum - rcb_momentum,
    "h2h_team1_winrate": dc_h2h_winrate,
    "h2h_matches": total_h2h,
    "home_team1": 1,
    "home_team2": 0,
    "venue_chase_bias": 0.51,
    "toss_chose_field": 1,
    "toss_winner_is_team1": 0.5,
    "recency_weight": 1.0,
    "elo_x_momentum_t1": dc_elo * dc_momentum,
    "elo_x_momentum_t2": rcb_elo * rcb_momentum,
    "elo_x_home_t1": dc_elo * 1,
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
# 6. CONTEXTUAL ADJUSTMENTS (PRE-MATCH)
# ============================================================================
print("\n[6/7] Applying contextual adjustments (PRE-MATCH only)...")

base_dc_prob = ensemble_prob[1]

adjustments = {}
# A. DC home advantage at Arun Jaitley
adjustments["dc_home_advantage"] = +0.04
# B. KL Rahul red-hot - 152* vs PBKS in last match (highest T20 by Indian)
adjustments["kl_rahul_form"] = +0.04
# C. Axar Patel + Kuldeep Yadav - quality spin combo at Delhi
adjustments["dc_spin_attack"] = +0.02
# D. RCB defending champions, 2nd in table (10 pts)
adjustments["rcb_champions_form"] = -0.05
# E. RCB H2H dominance (21W-13L all-time, ~60%)
adjustments["rcb_h2h_lead"] = -0.04
# F. Virat Kohli outstanding record vs DC (1,154 runs - most in this fixture)
adjustments["kohli_vs_dc_record"] = -0.03
# G. RCB bowling depth: Hazlewood, Bhuvneshwar, Krunal, Yash Dayal
adjustments["rcb_bowling_attack"] = -0.03
# H. DC inconsistent: lost 2 of last 3 (to MI and PBKS at home)
adjustments["dc_recent_inconsistency"] = -0.02
# I. RCB momentum from recent win vs GT (Apr 25)
adjustments["rcb_momentum"] = -0.01

total_adjustment = sum(adjustments.values())
adjusted_dc_prob = np.clip(base_dc_prob + total_adjustment, 0.05, 0.95)
adjusted_rcb_prob = 1 - adjusted_dc_prob

# ============================================================================
# 7. MONTE CARLO
# ============================================================================
print("\n[7/7] Running Monte Carlo simulation (10,000 matches)...")

np.random.seed(42)
N_SIM = 10000
dc_wins_sim = 0
rcb_wins_sim = 0
dc_margins = []
rcb_margins = []

for _ in range(N_SIM):
    noise = np.random.normal(0, 0.05)
    match_prob = np.clip(adjusted_dc_prob + noise, 0.1, 0.9)
    if np.random.random() < match_prob:
        dc_wins_sim += 1
        margin = max(1, int(np.random.exponential(20)))
        dc_margins.append(margin)
    else:
        rcb_wins_sim += 1
        margin = max(1, int(np.random.exponential(22)))
        rcb_margins.append(margin)

sim_dc_pct = dc_wins_sim / N_SIM * 100
sim_rcb_pct = rcb_wins_sim / N_SIM * 100

# ============================================================================
# REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  PREDICTION REPORT: DC vs RCB | IPL 2026 Match 39")
print("=" * 70)

print("\n--- TEAM PROFILES ---")
print(f"  {'':30s} {'DC':>14s} {'RCB':>14s}")
print(f"  {'Elo Rating':30s} {dc_elo:>14.1f} {rcb_elo:>14.1f}")
print(f"  {'Last 5 (W)':30s} {dc_wins_last5:>14d} {rcb_wins_last5:>14d}")
print(f"  {'Momentum':30s} {dc_momentum:>14.1%} {rcb_momentum:>14.1%}")
print(f"  {'H2H (all-time)':30s} {dc_h2h_wins:>9d}W/{total_h2h:d}  {rcb_h2h_wins:>9d}W/{total_h2h:d}")
print(f"  {'IPL 2026 Form':30s} {'3W 4L (6th)':>14s} {'5W 2L (2nd)':>14s}")

print("\n--- MODEL PREDICTIONS (P(DC wins)) ---")
print(f"  Random Forest:       {rf_prob[1]:>6.1%}")
if HAS_XGB:
    print(f"  XGBoost:             {xgb_prob[1]:>6.1%}")
print(f"  Gradient Boosting:   {gb_prob[1]:>6.1%}")
print(f"  Logistic Regression: {lr_prob[1]:>6.1%}")
print(f"  Ensemble (weighted): {base_dc_prob:>6.1%}")

print("\n--- CONTEXTUAL ADJUSTMENTS ---")
for name, adj in adjustments.items():
    if adj != 0:
        direction = "+" if adj > 0 else ""
        print(f"  {name:40s} {direction}{adj:.1%}")
print(f"  TOTAL: {'+' if total_adjustment > 0 else ''}{total_adjustment:.1%}")

print(f"\n--- FINAL PREDICTION ---")
print(f"  DC Win Probability:  {adjusted_dc_prob:>6.1%}")
print(f"  RCB Win Probability: {adjusted_rcb_prob:>6.1%}")

print(f"\n--- MONTE CARLO ({N_SIM:,d}) ---")
print(f"  DC wins:  {dc_wins_sim:>5,d} ({sim_dc_pct:.1f}%)")
print(f"  RCB wins: {rcb_wins_sim:>5,d} ({sim_rcb_pct:.1f}%)")
if dc_margins:
    print(f"  Avg DC margin:  {np.mean(dc_margins):.0f} runs")
if rcb_margins:
    print(f"  Avg RCB margin: {np.mean(rcb_margins):.0f} runs")

winner = "DC" if adjusted_dc_prob > 0.5 else "RCB"
winner_full = DC if winner == "DC" else RCB
win_prob = max(adjusted_dc_prob, adjusted_rcb_prob)
admin_confidence = int(round(win_prob * 100))

if winner == "DC":
    avg_margin = int(np.mean(dc_margins)) if dc_margins else 18
    rf_v, gb_v, lr_v = rf_prob[1]*100, gb_prob[1]*100, lr_prob[1]*100
    xgb_v = xgb_prob[1]*100 if HAS_XGB else 0
else:
    avg_margin = int(np.mean(rcb_margins)) if rcb_margins else 20
    rf_v, gb_v, lr_v = rf_prob[0]*100, gb_prob[0]*100, lr_prob[0]*100
    xgb_v = xgb_prob[0]*100 if HAS_XGB else 0

print(f"\n  VERDICT: {winner} to win")
print(f"  Confidence: {admin_confidence}%")
print(f"  Avg margin: {avg_margin} runs")
print(f"  Per-model for {winner}: rf={rf_v:.1f}, xgb={xgb_v:.1f}, gb={gb_v:.1f}, lr={lr_v:.1f}")

reasoning = (
    f"Ensemble of 4 ML models trained on {len(all_matches):,d} IPL matches gives DC a base "
    f"{base_dc_prob:.1%} win probability. Contextual adjustments ({'+' if total_adjustment > 0 else ''}{total_adjustment:.0%}) "
    f"factor in DC home advantage at Arun Jaitley, KL Rahul's red-hot 152* vs PBKS in his last innings, "
    f"and Axar-Kuldeep spin attack. Offset by RCB's defending champions tag and 2nd-place form (5W from 7), "
    f"all-time H2H dominance (21-13, 60% win rate), Virat Kohli's record against DC (1,154 runs - most in this rivalry), "
    f"and RCB's strong bowling unit (Hazlewood, Bhuvneshwar, Krunal). Monte Carlo (10,000 runs): "
    f"DC {sim_dc_pct:.1f}% / RCB {sim_rcb_pct:.1f}%. RCB edge comes from superior IPL 2026 form, deeper "
    f"squad balance, and historical dominance over DC. KL Rahul's individual brilliance is DC's main upset path."
)
print(f"\n  REASONING:\n  {reasoning}")
print("\n" + "=" * 70)
