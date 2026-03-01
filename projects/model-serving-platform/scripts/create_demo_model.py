"""Create a demo Iris model for local development.

Works from any working directory - always saves relative to project root.

Run: python scripts/create_demo_model.py
"""
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# Resolve project root from this script's location (scripts/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "models", "model.joblib")

if os.path.exists(MODEL_PATH):
    print(f"Model already exists at {MODEL_PATH}, skipping.")
else:
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Demo model saved to {MODEL_PATH}")

# Verify
print(f"[OK] Model ready at {MODEL_PATH}")
