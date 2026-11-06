# The learner notebook labs/lltLoRA/lltLoRA.ipynb, pushed unchanged as Kaggle kernel llt-lora-notebook. Cell timings come from the
# timestamps in the kernel log. Build the notebook first: python ../labs/lltLoRA/build_notebook.py
import json, os
_nb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltLoRA", "lltLoRA.ipynb"), encoding="utf-8"))
CELLS = [("md" if c["cell_type"] == "markdown" else "code", c["source"]) for c in _nb["cells"]]
