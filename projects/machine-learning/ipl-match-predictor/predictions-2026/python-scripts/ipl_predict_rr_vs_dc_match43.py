"""
IPL 2026 Match Prediction: RR vs DC (Match 43, May 1, 2026)
================================================================
Sawai Mansingh Stadium, Jaipur | 7:30 PM IST
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
print("  IPL 2026 PREDICTION: RR vs DC")
print("  Match 43 | Sawai Mansingh Stadium, Jaipur")
print("  May 1, 2026 | 7:30 PM IST")
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
# 2. SUPPLEMENT WITH 2025 + 2026 SEASON DATA (PRE-MATCH ONLY, before May 1)
# ============================================================================
print("\n[2/7] Supplementing with 2025 + 2026 season data...")

ipl_supp = [
    # RR 2025 results - finished 4th, made playoffs ~9W 5L
    {"date": "2025-03-23", "team1": "Sunrisers Hyderabad", "team2": "Rajasthan Royals", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "runs", "result_margin": 30},
    {"date": "2025-03-30", "team1": "Rajasthan Royals", "team2": "Chennai Super Kings", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "runs", "result_margin": 17},
    {"date": "2025-04-05", "team1": "Rajasthan Royals", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-09", "team1": "Royal Challengers Bangalore", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-13", "team1": "Rajasthan Royals", "team2": "Lucknow Super Giants", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "runs", "result_margin": 18},
    {"date": "2025-04-18", "team1": "Delhi Capitals", "team2": "Rajasthan Royals", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-22", "team1": "Rajasthan Royals", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "field", "result": "runs", "result_margin": 9},
    {"date": "2025-04-26", "team1": "Gujarat Titans", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "wickets", "result_margin": 7},
    {"date": "2025-05-01", "team1": "Rajasthan Royals", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "runs", "result_margin": 9},
    {"date": "2025-05-05", "team1": "Kolkata Knight Riders", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-10", "team1": "Rajasthan Royals", "team2": "Sunrisers Hyderabad", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "runs", "result_margin": 12},
    {"date": "2025-05-18", "team1": "Rajasthan Royals", "team2": "Sunrisers Hyderabad", "winner": "Sunrisers Hyderabad", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 6},

    # DC 2025 results - 5th place, missed playoffs ~7W 7L
    {"date": "2025-03-24", "team1": "Lucknow Super Giants", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "wickets", "result_margin": 5},
    {"date": "2025-03-30", "team1": "Sunrisers Hyderabad", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-03", "team1": "Chennai Super Kings", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "bat", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-08", "team1": "Delhi Capitals", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-13", "team1": "Mumbai Indians", "team2": "Delhi Capitals", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "runs", "result_margin": 12},
    {"date": "2025-04-18", "team1": "Delhi Capitals", "team2": "Rajasthan Royals", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-23", "team1": "Delhi Capitals", "team2": "Mumbai Indians", "winner": "Mumbai Indians", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-29", "team1": "Punjab Kings", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-05-05", "team1": "Delhi Capitals", "team2": "Kolkata Knight Riders", "winner": "Kolkata Knight Riders", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Kolkata Knight Riders", "toss_decision": "field", "result": "runs", "result_margin": 14},
    {"date": "2025-05-12", "team1": "Delhi Capitals", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 7},

    # IPL 2026 RR results (6W 3L before match 43) - first 4 wins, then 2W 3L
    {"date": "2026-03-29", "team1": "Rajasthan Royals", "team2": "Lucknow Super Giants", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "runs", "result_margin": 13},
    {"date": "2026-04-03", "team1": "Rajasthan Royals", "team2": "Gujarat Titans", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 14},
    {"date": "2026-04-06", "team1": "Chennai Super Kings", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "bat", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-10", "team1": "Rajasthan Royals", "team2": "Royal Challengers Bangalore", "winner": "Rajasthan Royals", "city": "Guwahati", "venue": "Barsapara Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 14},
    {"date": "2026-04-14", "team1": "Kolkata Knight Riders", "team2": "Rajasthan Royals", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 11},
    {"date": "2026-04-18", "team1": "Mumbai Indians", "team2": "Rajasthan Royals", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 9},
    {"date": "2026-04-22", "team1": "Rajasthan Royals", "team2": "Sunrisers Hyderabad", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "runs", "result_margin": 17},
    {"date": "2026-04-25", "team1": "Rajasthan Royals", "team2": "Sunrisers Hyderabad", "winner": "Sunrisers Hyderabad", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "wickets", "result_margin": 5},
    {"date": "2026-04-28", "team1": "Punjab Kings", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "wickets", "result_margin": 6},

    # IPL 2026 DC results (3W 5L before match 43) - 3-match losing streak, bowled out for 75 vs RCB
    {"date": "2026-03-30", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 7},
    {"date": "2026-04-02", "team1": "Royal Challengers Bangalore", "team2": "Delhi Capitals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-07", "team1": "Gujarat Titans", "team2": "Delhi Capitals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "runs", "result_margin": 19},
    {"date": "2026-04-11", "team1": "Mumbai Indians", "team2": "Delhi Capitals", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2026-04-16", "team1": "Delhi Capitals", "team2": "Chennai Super Kings", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "runs", "result_margin": 16},
    {"date": "2026-04-20", "team1": "Delhi Capitals", "team2": "Punjab Kings", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "runs", "result_margin": 12},
    {"date": "2026-04-24", "team1": "Kolkata Knight Riders", "team2": "Delhi Capitals", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "runs", "result_margin": 21},
    # DC 75 all out vs RCB - 9 wkts loss
    {"date": "2026-04-29", "team1": "Delhi Capitals", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 9},
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
    "Wankhede Stadium": 0.55,
    "Eden Gardens": 0.48,
    "MA Chidambaram Stadium": 0.45,
    "Rajiv Gandhi International Stadium": 0.50,
    "Narendra Modi Stadium": 0.47,
    "Arun Jaitley Stadium": 0.51,
    "Sawai Mansingh Stadium": 0.50,
    "BRSABV Ekana Cricket Stadium": 0.48,
    "Barsapara Cricket Stadium": 0.49,
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
# 5. PREDICT RR vs DC
# ============================================================================
print("\n[5/7] Predicting RR vs DC...")

RR = "Rajasthan Royals"
DC = "Delhi Capitals"

rr_elo = final_elo[RR]
dc_elo = final_elo[DC]

rr_matches = all_matches[(all_matches["team1"] == RR) | (all_matches["team2"] == RR)].tail(5)
dc_matches = all_matches[(all_matches["team1"] == DC) | (all_matches["team2"] == DC)].tail(5)
rr_wins_last5 = sum(rr_matches["winner"] == RR)
dc_wins_last5 = sum(dc_matches["winner"] == DC)
rr_momentum = rr_wins_last5 / 5
dc_momentum = dc_wins_last5 / 5

h2h_df = all_matches[
    ((all_matches["team1"] == RR) & (all_matches["team2"] == DC)) |
    ((all_matches["team1"] == DC) & (all_matches["team2"] == RR))
]
total_h2h = len(h2h_df[h2h_df["winner"].isin([RR, DC])])
rr_h2h_wins = sum(h2h_df["winner"] == RR)
dc_h2h_wins = total_h2h - rr_h2h_wins
rr_h2h_winrate = rr_h2h_wins / total_h2h if total_h2h > 0 else 0.5

# RR team1 (home at Sawai Mansingh Stadium Jaipur), DC team2
match_features = {
    "elo_team1": rr_elo,
    "elo_team2": dc_elo,
    "elo_diff": rr_elo - dc_elo,
    "momentum_team1": rr_momentum,
    "momentum_team2": dc_momentum,
    "momentum_diff": rr_momentum - dc_momentum,
    "h2h_team1_winrate": rr_h2h_winrate,
    "h2h_matches": total_h2h,
    "home_team1": 1,
    "home_team2": 0,
    "venue_chase_bias": 0.50,
    "toss_chose_field": 1,
    "toss_winner_is_team1": 0.5,
    "recency_weight": 1.0,
    "elo_x_momentum_t1": rr_elo * rr_momentum,
    "elo_x_momentum_t2": dc_elo * dc_momentum,
    "elo_x_home_t1": rr_elo * 1,
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

base_rr_prob = ensemble_prob[1]

adjustments = {}
# Pro-RR
# A. Sawai Mansingh fortress vs DC: RR 5-2 H2H at Jaipur, DC last won here in 2019
adjustments["rr_jaipur_fortress_vs_dc"] = +0.07
# B. RR coming off massive PBKS chase win (223 chased at Mullanpur Apr 28)
adjustments["rr_pbks_chase_momentum"] = +0.04
# C. Sooryavanshi 400 runs in 9 games (incl century + 2 fifties)
adjustments["sooryavanshi_form"] = +0.03
# D. Jofra Archer 14 wickets (top RR wicket-taker), Boult new ball
adjustments["rr_pace_attack"] = +0.02
# E. DC powerplay collapse (joint-fewest 7 wkts in PP all season)
adjustments["dc_powerplay_weakness"] = +0.02
# F. DC bowled out for 75 vs RCB last game (devastating collapse)
adjustments["dc_collapse_75_ao"] = +0.03
# G. DC on 3-match losing streak (L-L-L)
adjustments["dc_losing_streak"] = +0.03

# Anti-RR / Pro-DC
# H. Mitchell Starc returns from shoulder injury - boosts DC bowling
adjustments["starc_return_boost"] = -0.03
# I. DC squad has KL Rahul/Kuldeep/Axar - quality talent capable of upset
adjustments["dc_individual_class"] = -0.02
# J. Long all-time H2H even at 15-15 (or DC 16-15) suggests parity
adjustments["dc_h2h_parity"] = -0.02

total_adjustment = sum(adjustments.values())
adjusted_rr_prob = np.clip(base_rr_prob + total_adjustment, 0.05, 0.95)
adjusted_dc_prob = 1 - adjusted_rr_prob

# ============================================================================
# 7. MONTE CARLO
# ============================================================================
print("\n[7/7] Running Monte Carlo simulation (10,000 matches)...")

np.random.seed(42)
N_SIM = 10000
rr_wins_sim = 0
dc_wins_sim = 0
rr_margins = []
dc_margins = []

for _ in range(N_SIM):
    noise = np.random.normal(0, 0.05)
    match_prob = np.clip(adjusted_rr_prob + noise, 0.1, 0.9)
    if np.random.random() < match_prob:
        rr_wins_sim += 1
        margin = max(1, int(np.random.exponential(22)))
        rr_margins.append(margin)
    else:
        dc_wins_sim += 1
        margin = max(1, int(np.random.exponential(18)))
        dc_margins.append(margin)

sim_rr_pct = rr_wins_sim / N_SIM * 100
sim_dc_pct = dc_wins_sim / N_SIM * 100

# ============================================================================
# REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  PREDICTION REPORT: RR vs DC | IPL 2026 Match 43")
print("=" * 70)

print("\n--- TEAM PROFILES ---")
print(f"  {'':30s} {'RR':>14s} {'DC':>14s}")
print(f"  {'Elo Rating':30s} {rr_elo:>14.1f} {dc_elo:>14.1f}")
print(f"  {'Last 5 (W)':30s} {rr_wins_last5:>14d} {dc_wins_last5:>14d}")
print(f"  {'Momentum':30s} {rr_momentum:>14.1%} {dc_momentum:>14.1%}")
print(f"  {'H2H (all-time)':30s} {rr_h2h_wins:>9d}W/{total_h2h:d}  {dc_h2h_wins:>9d}W/{total_h2h:d}")
print(f"  {'IPL 2026 Form':30s} {'6W 3L (4th)':>14s} {'3W 5L (7th)':>14s}")

print("\n--- MODEL PREDICTIONS (P(RR wins)) ---")
print(f"  Random Forest:       {rf_prob[1]:>6.1%}")
if HAS_XGB:
    print(f"  XGBoost:             {xgb_prob[1]:>6.1%}")
print(f"  Gradient Boosting:   {gb_prob[1]:>6.1%}")
print(f"  Logistic Regression: {lr_prob[1]:>6.1%}")
print(f"  Ensemble (weighted): {base_rr_prob:>6.1%}")

print("\n--- CONTEXTUAL ADJUSTMENTS ---")
for name, adj in adjustments.items():
    if adj != 0:
        direction = "+" if adj > 0 else ""
        print(f"  {name:40s} {direction}{adj:.1%}")
print(f"  TOTAL: {'+' if total_adjustment > 0 else ''}{total_adjustment:.1%}")

print(f"\n--- FINAL PREDICTION ---")
print(f"  RR Win Probability:  {adjusted_rr_prob:>6.1%}")
print(f"  DC Win Probability:  {adjusted_dc_prob:>6.1%}")

print(f"\n--- MONTE CARLO ({N_SIM:,d}) ---")
print(f"  RR wins:  {rr_wins_sim:>5,d} ({sim_rr_pct:.1f}%)")
print(f"  DC wins:  {dc_wins_sim:>5,d} ({sim_dc_pct:.1f}%)")
if rr_margins:
    print(f"  Avg RR margin:  {np.mean(rr_margins):.0f} runs")
if dc_margins:
    print(f"  Avg DC margin:  {np.mean(dc_margins):.0f} runs")

winner = "RR" if adjusted_rr_prob > 0.5 else "DC"
winner_full = RR if winner == "RR" else DC
win_prob = max(adjusted_rr_prob, adjusted_dc_prob)
admin_confidence = int(round(win_prob * 100))

if winner == "RR":
    avg_margin = int(np.mean(rr_margins)) if rr_margins else 22
    rf_v, gb_v, lr_v = rf_prob[1]*100, gb_prob[1]*100, lr_prob[1]*100
    xgb_v = xgb_prob[1]*100 if HAS_XGB else 0
else:
    avg_margin = int(np.mean(dc_margins)) if dc_margins else 18
    rf_v, gb_v, lr_v = rf_prob[0]*100, gb_prob[0]*100, lr_prob[0]*100
    xgb_v = xgb_prob[0]*100 if HAS_XGB else 0

print(f"\n  VERDICT: {winner} to win")
print(f"  Confidence: {admin_confidence}%")
print(f"  Avg margin: {avg_margin} runs")
print(f"  Per-model for {winner}: rf={rf_v:.1f}, xgb={xgb_v:.1f}, gb={gb_v:.1f}, lr={lr_v:.1f}")

reasoning = (
    f"Ensemble of 4 ML models trained on {len(all_matches):,d} IPL matches gives RR a base "
    f"{base_rr_prob:.1%} win probability. Contextual adjustments ({'+' if total_adjustment > 0 else ''}{total_adjustment:.0%}) "
    f"factor in RR's Sawai Mansingh fortress (5-2 H2H vs DC at Jaipur, DC last won here in 2019), "
    f"momentum from RR's stunning chase of 223 vs PBKS (Apr 28, ending PBKS unbeaten run), "
    f"Vaibhav Sooryavanshi's 400 runs in 9 games (century + 2 fifties), and Jofra Archer's 14 wickets. "
    f"DC enter on a 3-match losing streak after being bowled out for just 75 vs RCB (9-wkt loss), "
    f"with the joint-fewest powerplay wickets in the season. Counter-factors: Mitchell Starc returns "
    f"from injury to bolster DC's pace attack, KL Rahul/Kuldeep/Axar provide individual class, and "
    f"the all-time H2H is dead even (15-15 or DC 16-15). Monte Carlo (10,000 runs): "
    f"RR {sim_rr_pct:.1f}% / DC {sim_dc_pct:.1f}%. RR's home dominance at Jaipur combined with DC's "
    f"complete collapse in their last outing makes this heavily lopsided. Starc's first match back is the upset path."
)
print(f"\n  REASONING:\n  {reasoning}")
print("\n" + "=" * 70)
