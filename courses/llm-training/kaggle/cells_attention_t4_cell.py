# The lab page's T4 cell (labs/lltAttention/cells/step_t4.py), run as shipped, plus a save of its output.
import os
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltAttention", "cells", "step_t4.py"), encoding="utf-8").read()
CELLS = [
("md", "# LLM Training Lab: lltAttention T4 cell\n\nThe page's step that compares naive attention with SDPA on a T4, run exactly as shipped."),
("code", _src),
]
