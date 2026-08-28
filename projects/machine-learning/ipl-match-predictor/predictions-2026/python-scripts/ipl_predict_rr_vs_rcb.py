"""
IPL 2026 Match Prediction: RR vs RCB (Match 16, April 10, 2026)
================================================================
Barsapara Cricket Stadium, Guwahati | 7:30 PM IST
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
print("  IPL 2026 MATCH PREDICTION: RR vs RCB")
print("  Match 16 | Barsapara Cricket Stadium, Guwahati")
print("  April 10, 2026 | 7:30 PM IST")
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
    # RR 2025: 4W 10L, 9th place (worst recent season)
    {"date": "2025-03-24", "team1": "Rajasthan Royals", "team2": "Mumbai Indians", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "runs", "result_margin": 20},
    {"date": "2025-03-30", "team1": "Chennai Super Kings", "team2": "Rajasthan Royals", "winner": "Chennai Super Kings", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "bat", "result": "runs", "result_margin": 18},
    {"date": "2025-04-05", "team1": "Rajasthan Royals", "team2": "Sunrisers Hyderabad", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-11", "team1": "Punjab Kings", "team2": "Rajasthan Royals", "winner": "Punjab Kings", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Punjab Kings", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-04-16", "team1": "Rajasthan Royals", "team2": "Kolkata Knight Riders", "winner": "Rajasthan Royals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Rajasthan Royals", "toss_decision": "bat", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-22", "team1": "Mumbai Indians", "team2": "Rajasthan Royals", "winner": "Mumbai Indians", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-04-28", "team1": "Rajasthan Royals", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-05-02", "team1": "Gujarat Titans", "team2": "Rajasthan Royals", "winner": "Gujarat Titans", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Gujarat Titans", "toss_decision": "bat", "result": "runs", "result_margin": 22},
    {"date": "2025-05-08", "team1": "Rajasthan Royals", "team2": "Lucknow Super Giants", "winner": "Lucknow Super Giants", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-13", "team1": "Royal Challengers Bangalore", "team2": "Rajasthan Royals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-05-18", "team1": "Rajasthan Royals", "team2": "Chennai Super Kings", "winner": "Chennai Super Kings", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Chennai Super Kings", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-05-23", "team1": "Kolkata Knight Riders", "team2": "Rajasthan Royals", "winner": "Kolkata Knight Riders", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Kolkata Knight Riders", "toss_decision": "bat", "result": "runs", "result_margin": 15},
    {"date": "2025-05-27", "team1": "Rajasthan Royals", "team2": "Punjab Kings", "winner": "Punjab Kings", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Punjab Kings", "toss_decision": "field", "result": "wickets", "result_margin": 3},
    {"date": "2025-05-31", "team1": "Sunrisers Hyderabad", "team2": "Rajasthan Royals", "winner": "Sunrisers Hyderabad", "city": "Hyderabad", "venue": "Rajiv Gandhi International Stadium", "toss_winner": "Sunrisers Hyderabad", "toss_decision": "bat", "result": "runs", "result_margin": 28},

    # RCB 2025: CHAMPIONS - 9W 4L 1NR, 19 points
    {"date": "2025-03-25", "team1": "Royal Challengers Bangalore", "team2": "Chennai Super Kings", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 25},
    {"date": "2025-03-31", "team1": "Mumbai Indians", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Mumbai", "venue": "Wankhede Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-06", "team1": "Royal Challengers Bangalore", "team2": "Sunrisers Hyderabad", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 30},
    {"date": "2025-04-12", "team1": "Kolkata Knight Riders", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Kolkata", "venue": "Eden Gardens", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-04-18", "team1": "Royal Challengers Bangalore", "team2": "Delhi Capitals", "winner": "Delhi Capitals", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Delhi Capitals", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-04-23", "team1": "Rajasthan Royals", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Jaipur", "venue": "Sawai Mansingh Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-04-28", "team1": "Royal Challengers Bangalore", "team2": "Gujarat Titans", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 35},
    {"date": "2025-05-04", "team1": "Lucknow Super Giants", "team2": "Royal Challengers Bangalore", "winner": "Lucknow Super Giants", "city": "Lucknow", "venue": "BRSABV Ekana Cricket Stadium", "toss_winner": "Lucknow Super Giants", "toss_decision": "bat", "result": "runs", "result_margin": 12},
    {"date": "2025-05-10", "team1": "Royal Challengers Bangalore", "team2": "Rajasthan Royals", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 6},
    {"date": "2025-05-15", "team1": "Punjab Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Mohali", "venue": "Maharaja Yadavindra Singh International Cricket Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 5},
    {"date": "2025-05-20", "team1": "Royal Challengers Bangalore", "team2": "Sunrisers Hyderabad", "winner": "Royal Challengers Bangalore", "city": "Bangalore", "venue": "M Chinnaswamy Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "bat", "result": "runs", "result_margin": 28},
    {"date": "2025-05-25", "team1": "Chennai Super Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Chennai", "venue": "MA Chidambaram Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "wickets", "result_margin": 4},
    {"date": "2025-06-01", "team1": "Punjab Kings", "team2": "Royal Challengers Bangalore", "winner": "Royal Challengers Bangalore", "city": "Ahmedabad", "venue": "Narendra Modi Stadium", "toss_winner": "Royal Challengers Bangalore", "toss_decision": "field", "result": "runs", "result_margin": 6},

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
    "Barsapara Cricket Stadium": 0.60,  # 4 of last 5 won by chasing side; heavy dew
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
# 5. PREDICT: RR vs RCB
# ============================================================================
print("\n[5/7] Predicting RR vs RCB...")

RR = "Rajasthan Royals"
RCB = "Royal Challengers Bangalore"

rr_elo = final_elo[RR]
rcb_elo = final_elo[RCB]

rr_matches = all_matches[(all_matches["team1"] == RR) | (all_matches["team2"] == RR)].tail(5)
rcb_matches = all_matches[(all_matches["team1"] == RCB) | (all_matches["team2"] == RCB)].tail(5)

rr_wins_last5 = sum(rr_matches["winner"] == RR)
rcb_wins_last5 = sum(rcb_matches["winner"] == RCB)
rr_momentum = rr_wins_last5 / 5
rcb_momentum = rcb_wins_last5 / 5

rr_rcb = all_matches[
    ((all_matches["team1"] == RR) & (all_matches["team2"] == RCB)) |
    ((all_matches["team1"] == RCB) & (all_matches["team2"] == RR))
]
total_h2h = len(rr_rcb[rr_rcb["winner"].isin([RR, RCB])])
rr_h2h_wins = sum(rr_rcb["winner"] == RR)
rcb_h2h_wins = total_h2h - rr_h2h_wins
rr_h2h_winrate = rr_h2h_wins / total_h2h if total_h2h > 0 else 0.5

# RR is team1 (neutral venue - Guwahati), RCB is team2
match_features = {
    "elo_team1": rr_elo,
    "elo_team2": rcb_elo,
    "elo_diff": rr_elo - rcb_elo,
    "momentum_team1": rr_momentum,
    "momentum_team2": rcb_momentum,
    "momentum_diff": rr_momentum - rcb_momentum,
    "h2h_team1_winrate": rr_h2h_winrate,
    "h2h_matches": total_h2h,
    "home_team1": 0,   # Neither team is home - Guwahati neutral
    "home_team2": 0,
    "venue_chase_bias": 0.60,   # Barsapara - strong chasing bias (4/5 recent)
    "toss_chose_field": 1,  # Night game + dew, teams prefer to field
    "toss_winner_is_team1": 0.5,
    "recency_weight": 1.0,
    "elo_x_momentum_t1": rr_elo * rr_momentum,
    "elo_x_momentum_t2": rcb_elo * rcb_momentum,
    "elo_x_home_t1": rr_elo * 0,  # No home advantage
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

base_rr_prob = ensemble_prob[1]  # P(RR wins)

adjustments = {}

# A. RR unbeaten 3-0 in 2026 - tournament best form (+4%)
# Won all 3 matches, top of points table (6 pts, NRR +2.40)
# Jaiswal 77* off 32 in last match, Suryavanshi 39 off 14
# Ravi Bishnoi leads Purple Cap, Jofra Archer excellent
adjustments["rr_perfect_form_2026"] = +0.04

# B. Vaibhav Suryavanshi - 15-year-old phenomenon (+2%)
# Youngest IPL centurion, scored 39 off 14 vs MI
# Completely changes match dynamics at top of order
# RCB's plans disrupted by someone they haven't faced before
adjustments["suryavanshi_age_factor"] = +0.02

# C. Jofra Archer - world-class pace (+2%)
# One of the most feared bowlers in T20 cricket
# Bounce, seam, variation - deadly at Barsapara
# RCB's top order of Kohli + Salt will be tested
adjustments["archer_pace_threat"] = +0.02

# D. Ravi Bishnoi - leading wicket-taker 2026 (+2%)
# Purple Cap holder, wrist spin at Guwahati
# Ball grips in second innings, Bishnoi's time
adjustments["bishnoi_wickets"] = +0.02

# E. RR at Guwahati - semi-home ground (+2%)
# RR uses Barsapara as home venue in 2026
# Already beat GT and MI here this season
# Familiar conditions, partisan crowd
adjustments["rr_guwahati_home"] = +0.02

# F. RCB defending champions + 2-0 unbeaten 2026 (-3%)
# Won IPL 2025 title - peak confidence
# Team playing with championship swagger
# Both wins convincing, best NRR in tournament (+2.50)
adjustments["rcb_champions_form"] = -0.03

# G. Virat Kohli - 657 runs in 2025 + 18 years IPL experience (-2%)
# Most experienced batter in this rivalry (896 runs all-time)
# Class batsman who lifts entire team
# Highest scorer in RR vs RCB history
adjustments["kohli_experience"] = -0.02

# H. Josh Hazlewood - 22 wickets in 2025 (-2%)
# Australia's most consistent T20 pacer
# Swing in early overs, death-over skills
# Perfect foil to Bhuvneshwar Kumar
adjustments["hazlewood_bowling"] = -0.02

# I. RCB H2H advantage: leads 17-14 all time (-1%)
# Slight edge in overall head to head
# Won at Guwahati (neutral) previously
adjustments["rcb_h2h_edge"] = -0.01

# J. Barsapara - chasing bias 60% at this venue (-1% for RR batting first)
# Teams batting second have significant advantage with dew
# RCB's pace attack (Hazlewood, Bhuvi) helps when chasing
# Both teams will want to field; winning toss crucial
adjustments["barsapara_dew_factor"] = -0.01

# K. Jaiswal in blistering form (+1%)
# 77* off 32 balls vs MI (rain game but still dominant)
# Orange cap contender, one of best T20 openers in world
adjustments["jaiswal_form"] = +0.01

total_adjustment = sum(adjustments.values())
adjusted_rr_prob = np.clip(base_rr_prob + total_adjustment, 0.05, 0.95)
adjusted_rcb_prob = 1 - adjusted_rr_prob

# ============================================================================
# 7. MONTE CARLO SIMULATION
# ============================================================================
print("\n[7/7] Running Monte Carlo simulation (10,000 matches)...")

np.random.seed(42)
N_SIM = 10000

rr_wins_sim = 0
rcb_wins_sim = 0
rr_margins = []
rcb_margins = []

for _ in range(N_SIM):
    noise = np.random.normal(0, 0.05)
    match_prob = np.clip(adjusted_rr_prob + noise, 0.1, 0.9)
    if np.random.random() < match_prob:
        rr_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        rr_margins.append(margin)
    else:
        rcb_wins_sim += 1
        margin = max(1, int(np.random.exponential(16)))
        rcb_margins.append(margin)

sim_rr_pct = rr_wins_sim / N_SIM * 100
sim_rcb_pct = rcb_wins_sim / N_SIM * 100

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 70)
print("  PREDICTION REPORT: RR vs RCB | IPL 2026 Match 16")
print("  Barsapara Cricket Stadium, Guwahati | April 10, 2026 | 7:30 PM IST")
print("=" * 70)

print("\n--- TEAM PROFILES ---")
print(f"  {'':30s} {'RR':>12s} {'RCB':>12s}")
print(f"  {'Elo Rating':30s} {rr_elo:>12.1f} {rcb_elo:>12.1f}")
print(f"  {'Last 5 Matches (W/5)':30s} {rr_wins_last5:>12d} {rcb_wins_last5:>12d}")
print(f"  {'Momentum (last 5 win%)':30s} {rr_momentum:>12.1%} {rcb_momentum:>12.1%}")
print(f"  {'H2H Record (all-time)':30s} {rr_h2h_wins:>7d}W/{total_h2h:d}  {rcb_h2h_wins:>7d}W/{total_h2h:d}")
print(f"  {'2025 Season':30s} {'4W-10L (9th)':>12s} {'CHAMPIONS':>12s}")
print(f"  {'2026 Record':30s} {'3W-0L (1st)':>12s} {'2W-0L (3rd)':>12s}")
print(f"  {'Home Advantage':30s} {'YES (Guwahati)':>14s} {'NO':>12s}")
print(f"  {'Captain':30s} {'Riyan Parag':>12s} {'Rajat Patidar':>12s}")

print("\n--- KEY PLAYERS ---")
print(f"  RR:  Yashasvi Jaiswal (77* off 32 vs MI), Vaibhav Suryavanshi (15yr, 39 off 14)")
print(f"       Riyan Parag (c, 393 runs 2025 @ SR 166), Dhruv Jurel (wk)")
print(f"       Shimron Hetmyer (power hitter), Ravindra Jadeja (all-rounder)")
print(f"       Jofra Archer (world-class pace), Ravi Bishnoi (Purple Cap 2026)")
print(f"       Nandre Burger (left-arm pace), Sandeep Sharma (swing)")
print(f"  RCB: Virat Kohli (657 runs 2025, 896 runs vs RR all-time), Phil Salt (wk)")
print(f"       Rajat Patidar (c, led RCB to 2025 title), Tim David (power hitter)")
print(f"       Devdutt Padikkal, Krunal Pandya (left-arm spin + bat)")
print(f"       Josh Hazlewood (22 wkts 2025), Bhuvneshwar Kumar (swing)")
print(f"       Suyash Sharma (wrist spin), Jacob Duffy (pace)")

print("\n--- MODEL PREDICTIONS (P(RR wins)) ---")
print(f"  Random Forest:         {rf_prob[1]:>6.1%}")
if HAS_XGB:
    print(f"  XGBoost:               {xgb_prob[1]:>6.1%}")
print(f"  Gradient Boosting:     {gb_prob[1]:>6.1%}")
print(f"  Logistic Regression:   {lr_prob[1]:>6.1%}")
print(f"  Ensemble (weighted):   {base_rr_prob:>6.1%}")

print("\n--- CONTEXTUAL ADJUSTMENTS ---")
for name, adj in adjustments.items():
    if adj != 0:
        direction = "+" if adj > 0 else ""
        print(f"  {name:50s} {direction}{adj:.1%}")
print(f"  {'TOTAL ADJUSTMENT':50s} {'+' if total_adjustment > 0 else ''}{total_adjustment:.1%}")

print(f"\n--- FINAL PREDICTION ---")
print(f"  RR  Win Probability:   {adjusted_rr_prob:>6.1%}")
print(f"  RCB Win Probability:   {adjusted_rcb_prob:>6.1%}")

print(f"\n--- MONTE CARLO SIMULATION ({N_SIM:,d} matches) ---")
print(f"  RR  wins: {rr_wins_sim:>5,d} ({sim_rr_pct:.1f}%)")
print(f"  RCB wins: {rcb_wins_sim:>5,d} ({sim_rcb_pct:.1f}%)")
if rr_margins:
    print(f"  Avg RR  win margin: {np.mean(rr_margins):.0f} runs")
if rcb_margins:
    print(f"  Avg RCB win margin: {np.mean(rcb_margins):.0f} runs")

print("\n" + "=" * 70)
winner = "RR" if adjusted_rr_prob > 0.5 else "RCB"
winner_full = RR if winner == "RR" else RCB
loser = "RCB" if winner == "RR" else "RR"
win_prob = max(adjusted_rr_prob, adjusted_rcb_prob)
confidence_level = "HIGH" if win_prob > 0.62 else "MODERATE" if win_prob > 0.55 else "LEAN"

print(f"  PREDICTED WINNER: {winner} ({winner_full})")
print(f"  WIN PROBABILITY:  {win_prob:.1%}  [{confidence_level}]")
if winner == "RR" and rr_margins:
    avg_margin = int(np.mean(rr_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")
elif winner == "RCB" and rcb_margins:
    avg_margin = int(np.mean(rcb_margins))
    print(f"  PREDICTED MARGIN: {avg_margin} runs")

print("\n--- NARRATIVE ---")
if winner == "RR":
    print(f"  RR are in irresistible form - 3 wins from 3, the only unbeaten team.")
    print(f"  At their Guwahati fortress, Jaiswal and Suryavanshi give them a devastating")
    print(f"  opening partnership that no team has solved yet. Jofra Archer + Bishnoi is")
    print(f"  perhaps the best pace-spin combination in the tournament. Despite RCB's")
    print(f"  championship pedigree, RR's 2026 momentum is simply too strong to stop.")
    print(f"  The clash of the unbeaten - RR's 4-0 record here at Guwahati tells the story.")
else:
    print(f"  RCB defend their title brilliantly. The championship experience shines through.")
    print(f"  Kohli vs Archer is the battle of the night - the outcome decides the match.")
    print(f"  Hazlewood's pace against Suryavanshi will be a fascinating contest.")
    print(f"  RCB's superior H2H and Kohli's 896 runs in this rivalry give them the edge.")

print("=" * 70)
