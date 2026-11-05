"""Check that the notebook's training cell (steps/nb_train.py) reproduces train_small.py.

Runs steps/nb_model.py, then steps/nb_train.py with its loop stopped after the step-250 evaluation
(the schedule still uses STEPS = 1500), and compares the logged val loss with run lltGeneration-t1-model-s0.

    ../../.venv-latest/Scripts/python.exe verify_nb_train.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {"__name__": "__main__"}
exec(compile(open(os.path.join(HERE, "steps", "nb_model.py"), encoding="utf-8").read(), "nb_model.py", "exec"), ns)
src = open(os.path.join(HERE, "steps", "nb_train.py"), encoding="utf-8").read()
assert "for step in range(STEPS + 1):" in src
src = src.replace("for step in range(STEPS + 1):", "for step in range(251):")
exec(compile(src, "nb_train.py", "exec"), ns)
ref = json.load(open(os.path.join(HERE, "..", "..", "runs", "lltGeneration", "lltGeneration-t1-model-s0.json"), encoding="utf-8"))
want = {m["step"]: m["valLoss"] for m in ref["metrics"]}
print("train_small.py val loss at 0 and 250:", want[0], want[250])
