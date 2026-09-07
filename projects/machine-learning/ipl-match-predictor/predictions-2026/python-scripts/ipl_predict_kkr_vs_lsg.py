"""
IPL 2026 Match Prediction: KKR vs LSG (Match 15, April 9, 2026)
================================================================
Eden Gardens, Kolkata | 7:30 PM IST
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
print("  IPL 2026 MATCH PREDICTION: KKR vs LSG")
print("  Match 15 | Eden Gardens, Kolkata")
print("  April 9, 2026 | 7:30 PM IST")
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
    # KKR 2025: 5W 9L, 8th place
    {"date": "2025-03-24", "team1": "Kolkata Knight Riders", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-03-30", "team1": "Kolkata Knight Riders", "team2": "Chennai Super Kings", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 22},
    {"date": "2025-04-05", "team1": "Punjab Kings", "team2": "Kolkata Knight Riders", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-04-10", "team1": "Kolkata Knight Riders", "team2": "Rajasthan Royals", "winner": "Rajasthan Royals", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-15", "team1": "Delhi Capitals", "team2": "Kolkata Knight Riders", "winner": "Kolkata Knight Riders", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Kolkata Knight Riders", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-20", "team1": "Kolkata Knight Riders", "team2": "Mumbai Indians", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2025-04-25", "team1": "Kolkata Knight Riders", "team2": "Sunrisers Hyderabad", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-05-01", "team1": "Gujarat Titans", "team2": "Kolkata Knight Riders", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-05-06", "team1": "Kolkata Knight Riders", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-12", "team1": "Rajasthan Royals", "team2": "Kolkata Knight Riders", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-17", "team1": "Kolkata Knight Riders", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "runs", "result_margin": 12},
    {"date": "2025-05-22", "team1": "Sunrisers Hyderabad", "team2": "Kolkata Knight Riders", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 22},
    {"date": "2025-05-26", "team1": "Kolkata Knight Riders", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-30", "team1": "Chennai Super Kings", "team2": "Kolkata Knight Riders", "winner": "Kolkata Knight Riders", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Kolkata Knight Riders", "toss_decision": "field", "result": "wickets", "result_margin": 3},

    # LSG 2025: 6W 8L, 7th place
    {"date": "2025-03-25", "team1": "Lucknow Super Giants", "team2": "Mumbai Indians", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 22},
    {"date": "2025-03-31", "team1": "Rajasthan Royals", "team2": "Lucknow Super Giants", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-06", "team1": "Lucknow Super Giants", "team2": "Chennai Super Kings", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2025-04-12", "team1": "Gujarat Titans", "team2": "Lucknow Super Giants", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-04-16", "team1": "Lucknow Super Giants", "team2": "Sunrisers Hyderabad", "winner": "Sunrisers Hyderabad", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-22", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-04-28", "team1": "Lucknow Super Giants", "team2": "Punjab Kings", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-05-04", "team1": "Kolkata Knight Riders", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-10", "team1": "Lucknow Super Giants", "team2": "Royal Challengers Bangalore", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 12},
    {"date": "2025-05-15", "team1": "Mumbai Indians", "team2": "Lucknow Super Giants", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 35},
    {"date": "2025-05-20", "team1": "Lucknow Super Giants", "team2": "Gujarat Titans", "winner": "Gujarat Titans", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "field", "result": "wickets", "result_margin": 3},
    {"date": "2025-05-24", "team1": "Chennai Super Kings", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-28", "team1": "Lucknow Super Giants", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 7},
    {"date": "2025-06-01", "team1": "Royal Challengers Bangalore", "team2": "Lucknow Super Giants", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 6},

    # IPL 2026 results so far (Matches 1-14)
    {"date": "2026-03-28", "team1": "Royal Challengers Bangalore", "team2": "Sunrisers Hyderabad", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-01", "team1": "Mumbai Indians", "team2": "Kolkata Knight Riders", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 17},
    {"date": "2026-04-02", "team1": "Rajasthan Royals", "team2": "Chennai Super Kings", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 17},
    {"date": "2026-04-03", "team1": "Rajasthan Royals", "team2": "Gujarat Titans", "winner": "Rajasthan Royals", "city": "Guwahati", "venue": "Barsapara Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2026-04-01", "team1": "Delhi Capitals", "team2": "Lucknow Super Giants", "winner": "Delhi Capitals", "city": "Delhi", "venue": "Arun Jaitley Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2026-04-02", "team1": "Sunrisers Hyderabad", "team2": "Kolkata Knight Riders", "winner": "Sunrisers Hyderabad", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 65},
    {"date": "2026-04-05", "team1": "Rajasthan Royals", "team2": "Delhi Capitals", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2026-04-05", "team1": "Sunrisers Hyderabad", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2026-04-05", "team1": "Royal Challengers Bangalore", "team2": "Chennai Super Kings", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "runs", "result_margin": 43},
    {"date": "2026-04-06", "team1": "Punjab Kings", "team2": "Kolkata Knight Riders", "winner": "No Result", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "no result", "result_margin": 0},
    {"date": "2026-04-07", "team1": "Rajasthan Royals", "team2": "Mumbai Indians", "winner": "Rajasthan Royals", "city": "Guwahati", "venue": "Barsapara Cricket Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 27},
    {"date": "2026-04-08", "team1": "Gujarat Titans", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 3},
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
    "Eden Gardens": 56 / 97,          # 57.7% - dew factor favors chasing
    "M Chinnaswamy Stadium": 53 / 98,
    "Wankhede Stadium": 0.52,
    "MA Chidambaram Stadium": 0.45,
    "Rajiv Gandhi International Stadium": 0.50,
    "Narendra Modi Stadium": 0.47,
    "Arun Jaitley Stadium": 0.51,
    "BRSABV Ekana Cricket Stadium": 11 / 20,
    "Maharaja Yadavindra Singh International Cricket Stadium": 7 / 17,
    "Sawai Mansingh Stadium": 0.50,
    "Barsapara Cricket Stadium": 0.60,   # 4 of last 5 won by chasing team
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
# 5. PREDICT: KKR vs LSG
# ============================================================================
print("\n[5/7] Predicting KKR vs LSG...")

KKR = "Kolkata Knight Riders"
LSG = "Lucknow Super Giants"

kkr_elo = final_elo[KKR]
lsg_elo = final_elo[LSG]

kkr_matches = all_matches[(all_matches["team1"] == KKR) | (all_matches["team2"] == KKR)].tail(5)
lsg_matches = all_matches[(all_matches["team1"] == LSG) | (all_matches["team2"] == LSG)].tail(5)

kkr_wins_last5 = sum(kkr_matches["winner"] == KKR)
lsg_wins_last5 = sum(lsg_matches["winner"] == LSG)
kkr_momentum = kkr_wins_last5 / 5
lsg_momentum = lsg_wins_last5 / 5

kkr_lsg = all_matches[
    ((all_matches["team1"] == KKR) & (all_matches["team2"] == LSG)) |
    ((all_matches["team1"] == LSG) & (all_matches["team2"] == KKR))
]
total_h2h = len(kkr_lsg[kkr_lsg["winner"].isin([KKR, LSG])])
kkr_h2h_wins = sum(kkr_lsg["winner"] == KKR)
lsg_h2h_wins = total_h2h - kkr_h2h_wins
kkr_h2h_winrate = kkr_h2h_wins / total_h2h if total_h2h > 0 else 0.5

# KKR team1 (home at Kolkata), LSG team2
match_features = {
    "elo_team1": kkr_elo,
    "elo_team2": lsg_elo,
    "elo_diff": kkr_elo - lsg_elo,
    "momentum_team1": kkr_momentum,
    "momentum_team2": lsg_momentum,
    "momentum_diff": kkr_momentum - lsg_momentum,
    "h2h_team1_winrate": kkr_h2h_winrate,
    "h2h_matches": total_h2h,
    "home_team1": 1,    # KKR at Eden Gardens, Kolkata
    "home_team2": 0,
    "venue_chase_bias": 56 / 97,   # Eden Gardens - dew factor, 57.7% chasing wins
    "toss_chose_field": 1,   # Night game - teams prefer field (dew)
    "toss_winner_is_team1": 0.5,
    "recency_weight": 1.0,
    "elo_x_momentum_t1": kkr_elo * kkr_momentum,
    "elo_x_momentum_t2": lsg_elo * lsg_momentum,
    "elo_x_home_t1": kkr_elo * 1,
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

base_kkr_prob = ensemble_prob[1]  # P(KKR wins)

adjustments = {}

# A. KKR home at Eden Gardens (+3%)
# Kolkata fortress, partisan crowd, 57% home win rate all-time
# IPL's most iconic venue
adjustments["kkr_home_eden_gardens"] = +0.03

# B. Sunil Narine confirmed available (+3%)
# Narine was missing in last game vs PBKS alongside Varun
# Narine is KKR's most impactful player - can single-handedly win games
# Opens batting + bowls crucial overs
adjustments["narine_confirmed"] = +0.03

# C. KKR desperate for first win - backs-against-wall motivation (+2%)
# 0W-2L-1NR, sitting last position effectively
# Eden Gardens crowd + desperation = big motivation factor
adjustments["kkr_backs_against_wall"] = +0.02

# D. Cameron Green batting depth (+1%)
# Rs. 25.20 crore acquisition, world-class all-rounder
# Adds batting depth and bowling options
adjustments["cameron_green_impact"] = +0.01

# E. Rinku Singh - Eden Gardens crowd favourite (+1%)
# Rinku's finishing ability is world-class
# Performs better under pressure with home crowd support
adjustments["rinku_home_support"] = +0.01

# F. KKR's 65-run loss to SRH - bowling vulnerabilities (-3%)
# SRH posted 226/8 and KKR folded for 161
# Exposed serious batting fragility and bowling without Narine+Varun
# LSG will have detailed plans
adjustments["kkr_exposed_vs_srh"] = -0.03

# G. Varun Chakravarthy doubtful (injury) (-2%)
# Bowled at training but decision at toss
# If absent, KKR's spin attack is severely weakened
# In 2019, this was the first time both Narine+Varun were unavailable
adjustments["varun_injury_risk"] = -0.02

# H. Rishabh Pant - x-factor for LSG (+2% for LSG = -2% for KKR)
# Rs. 27 crore, IPL's most expensive player
# 68* off 50 balls vs SRH - showed his class
# Leads from the front, unpredictable genius
adjustments["pant_xfactor"] = -0.02

# I. LSG beat SRH by 5 wickets - strong win (+1% for LSG = -1% for KKR)
# Showed LSG can handle top opposition
# Mohammed Shami 2/9 - exceptional spell
# Building confidence with 2nd match win
adjustments["lsg_momentum_win"] = -0.01

# J. H2H: LSG leads KKR 4-2 ALL TIME (-1%)
# LSG also lead at Eden Gardens 2-1 in 3 matches
# Historical pattern favors LSG
adjustments["lsg_h2h_advantage"] = -0.01

# K. Eden Gardens dew factor - helps batting team 2nd (-1%)
# Night game = significant dew after 8:30 PM
# Ball becomes slippery, spinners struggle
# LSG's pace-heavy attack benefits from batting 2nd
adjustments["dew_factor_chasing"] = -0.01

total_adjustment = sum(adjustments.values())
adjusted_kkr_prob = np.clip(base_kkr_prob + total_adjustment, 0.05, 0.95)
adjusted_lsg_prob = 1 - adjusted_kkr_prob

# ============================================================================
# 7. MONTE CARLO SIMULATION
# ============================================================================
print("\n[7/7] Running Monte Carlo simulation (10,000 matches)...")

np.random.seed(42)
N_SIM = 10000

kkr_wins_sim = 0
lsg_wins_sim = 0
kkr_margins = []
lsg_margins = []

for _ in range(N_SIM):
    noise = np.random.normal(0, 0.05)
    match_prob = np.clip(adjusted_kkr_prob + noise, 0.1, 0.9)
    if np.random.random() < match_prob:
        kkr_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        kkr_margins.append(margin)
    else:
        lsg_wins_sim += 1
        margin = max(1, int(np.random.exponential(18)))
        lsg_margins.append(margin)

sim_kkr_pct = kkr_wins_sim / N_SIM * 100
sim_lsg_pct = lsg_wins_sim / N_SIM * 100

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  PREDICTION REPORT: KKR vs LSG | IPL 2026 Match 15")
print("  Eden Gardens, Kolkata | April 9, 2026 | 7:30 PM IST")
print("=" * 70)

print("\n--- TEAM PROFILES ---")
print(f"  {'':30s} {'KKR':>12s} {'LSG':>12s}")
print(f"  {'Elo Rating':30s} {kkr_elo:>12.1f} {lsg_elo:>12.1f}")
print(f"  {'Last 5 Matches (W/5)':30s} {kkr_wins_last5:>12d} {lsg_wins_last5:>12d}")
print(f"  {'Momentum (last 5 win%)':30s} {kkr_momentum:>12.1%} {lsg_momentum:>12.1%}")
print(f"  {'H2H Record (all-time)':30s} {kkr_h2h_wins:>7d}W/{total_h2h:d}  {lsg_h2h_wins:>7d}W/{total_h2h:d}")
print(f"  {'2025 Season':30s} {'5W-9L (8th)':>12s} {'6W-8L (7th)':>12s}")
print(f"  {'2026 Record':30s} {'0W-2L-1NR':>12s} {'1W-1L':>12s}")
print(f"  {'Home Advantage':30s} {'YES (Eden)':>12s} {'NO':>12s}")
print(f"  {'Captain':30s} {'Ajinkya Rahane':>14s} {'Rishabh Pant':>12s}")

print("\n--- KEY PLAYERS ---")
print(f"  KKR: Sunil Narine (confirmed available - bat+bowl), Ajinkya Rahane (c)")
print(f"       Cameron Green (Rs 25.2cr, world-class all-rounder), Rinku Singh (finisher)")
print(f"       Angkrish Raghuvanshi (wk), Vaibhav Arora (17 wkts 2025)")
print(f"       Varun Chakravarthy (DOUBTFUL - finger injury, decision at toss)")
print(f"       Blessing Muzarabani (pace), Ramandeep Singh")
print(f"  LSG: Rishabh Pant (c/wk, 68* vs SRH, Rs 27cr), Mitchell Marsh (627 runs 2025)")
print(f"       Nicholas Pooran (H2H record vs KKR), Aiden Markram (SA captain)")
print(f"       Mohammed Shami (2/9 vs SRH, swing king), Mayank Yadav (150+ kph)")
print(f"       Avesh Khan (death bowling), Digvesh Rathi (14 wkts 2025)")

print("\n--- MODEL PREDICTIONS (P(KKR wins)) ---")
print(f"  Random Forest:         {rf_prob[1]:>6.1%}")
if HAS_XGB:
    print(f"  XGBoost:               {xgb_prob[1]:>6.1%}")
print(f"  Gradient Boosting:     {gb_prob[1]:>6.1%}")
print(f"  Logistic Regression:   {lr_prob[1]:>6.1%}")
print(f"  Ensemble (weighted):   {base_kkr_prob:>6.1%}")

print("\n--- CONTEXTUAL ADJUSTMENTS ---")
for name, adj in adjustments.items():
    if adj != 0:
        direction = "+" if adj > 0 else ""
        print(f"  {name:50s} {direction}{adj:.1%}")
print(f"  {'TOTAL ADJUSTMENT':50s} {'+' if total_adjustment > 0 else ''}{total_adjustment:.1%}")

print(f"\n--- FINAL PREDICTION ---")
print(f"  KKR Win Probability:   {adjusted_kkr_prob:>6.1%}")
print(f"  LSG Win Probability:   {adjusted_lsg_prob:>6.1%}")

print(f"\n--- MONTE CARLO SIMULATION ({N_SIM:,d} matches) ---")
print(f"  KKR wins: {kkr_wins_sim:>5,d} ({sim_kkr_pct:.1f}%)")
print(f"  LSG wins: {lsg_wins_sim:>5,d} ({sim_lsg_pct:.1f}%)")
if kkr_margins:
    print(f"  Avg KKR win margin: {np.mean(kkr_margins):.0f} runs")
if lsg_margins:
    print(f"  Avg LSG win margin: {np.mean(lsg_margins):.0f} runs")

print("\n" + "=" * 70)
winner = "KKR" if adjusted_kkr_prob > 0.5 else "LSG"
winner_full = KKR if winner == "KKR" else LSG
loser = "LSG" if winner == "KKR" else "KKR"
win_prob = max(adjusted_kkr_prob, adjusted_lsg_prob)
confidence_level = "HIGH" if win_prob > 0.62 else "MODERATE" if win_prob > 0.55 else "LEAN"

print(f"  PREDICTED WINNER: {winner} ({winner_full})")
print(f"  WIN PROBABILITY:  {win_prob:.1%}  [{confidence_level}]")
if winner == "KKR" and kkr_margins:
    avg_margin = int(np.mean(kkr_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")
elif winner == "LSG" and lsg_margins:
    avg_margin = int(np.mean(lsg_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")

print("\n--- NARRATIVE ---")
if winner == "LSG":
    print(f"  LSG have the edge at Eden Gardens tonight. Despite KKR's fortress reputation,")
    print(f"  LSG lead the H2H 4-2 overall and 2-1 at this ground. Rishabh Pant's explosive")
    print(f"  batting and Mohammed Shami's swing bowling give LSG a dangerous edge. KKR are")
    print(f"  winless and playing without Varun Chakravarthy (possibly). The dew factor under")
    print(f"  lights at Eden favors the chasing side - and LSG have the batting firepower to")
    print(f"  chase down any target. Narine's return helps KKR, but this is LSG's night.")
else:
    print(f"  KKR bounce back at home. Eden Gardens roars for them tonight. With Narine back,")
    print(f"  KKR have their most dangerous weapon restored. The fortress factor combined with")
    print(f"  KKR's desperation for a first win creates perfect storm conditions for a comeback.")
    print(f"  LSG showed good form vs SRH but face a very different beast at Eden Gardens.")

print("=" * 70)
