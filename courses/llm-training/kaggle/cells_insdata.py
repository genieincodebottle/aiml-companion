"""Kaggle kernel llt-insdata-notebook: the lltInstructionData lab notebook, run end to end on a T4.

The code cells are the lab page's cells (labs/lltInstructionData/cells/*.py) in order, after the runtime
check and the pinned install (labs/lltInstructionData/notebook_head.py). One extra cell at the end prints
the run's environment and copies the step outputs into insdata_notebook_record.json for the Run Library.
"""
import os, runpy

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltInstructionData")
head = runpy.run_path(os.path.join(LAB, "notebook_head.py"))
ORDER = ["c01_pool.py", "c02_teacher.py", "c03_dedup.py", "c04_decontam.py", "c05_pii.py", "c06_reward.py",
         "c07_select.py", "c08_train.py", "c09_eval.py", "c10_curated.py", "c11_bulk.py", "c12_recorded.py"]

RECORD = r'''
import json, platform, time
import transformers, trl, datasets, sentence_transformers
rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0),
       "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "transformers": transformers.__version__, "trl": trl.__version__, "datasets": datasets.__version__,
               "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__},
       "trainLogs": TRAIN_LOGS, "ifevalLoose": {k: v.tolist() for k, v in results.items()},
       "stepSeconds": STEP_SECONDS}
json.dump(rec, open("insdata_notebook_record.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != "ifevalLoose"}, indent=1)[:3000])
'''

TIMER = '''
import time as _t
STEP_SECONDS = globals().get("STEP_SECONDS", {})
_T0 = _t.time()
'''


def timed(name, src):
    return TIMER + src + "\nSTEP_SECONDS[" + repr(name) + "] = round(_t.time() - _T0, 1)\nprint('cell seconds', STEP_SECONDS[" + repr(name) + "])\n"


CELLS = [("md", "# lltInstructionData: the lab notebook, recorded run on a Kaggle T4\n\nThe page's cells in order. Each cell is wrapped in a timer for the Run Library; the learner notebook has no timers.")]
CELLS.append(("code", head["RUNTIME_CHECK"]))
CELLS.append(("code", head["INSTALL"]))
for f in ORDER:
    src = open(os.path.join(LAB, "cells", f), encoding="utf-8").read()
    CELLS.append(("md", "## " + f))
    CELLS.append(("code", timed(f, src)))
CELLS.append(("code", RECORD))
