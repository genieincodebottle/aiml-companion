"""Create a demo Iris model for local development.

Works from any working directory - always saves relative to project root.

Run: python scripts/create_demo_model.py
     python scripts/create_demo_model.py --force   # rebuild an existing model
"""
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# Resolve project root from this script's location (scripts/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "models", "model.joblib")

args = argparse.ArgumentParser(description=__doc__)
args.add_argument("--force", action="store_true",
                  help="rebuild even if a model file is already present")
opts = args.parse_args()

# A pickle is bound to the library version that wrote it. Loading a model
# pickled by a different scikit-learn raises InconsistentVersionWarning, and
# sklearn's own wording is "may lead to invalid results" -- silently wrong
# predictions, not a crash. Skipping unconditionally when a file exists meant a
# stale incompatible artifact could never be refreshed without deleting it by
# hand, so --force exists.
if os.path.exists(MODEL_PATH) and not opts.force:
    print(f"Model already exists at {MODEL_PATH}, skipping "
          f"(pass --force to rebuild; do this after upgrading scikit-learn).")
else:
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Demo model saved to {MODEL_PATH}")

# Verify
print(f"[OK] Model ready at {MODEL_PATH}")
