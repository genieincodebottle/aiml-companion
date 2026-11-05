"""lltGPT step 8 follow-up: post-norm and modern at learning rates 1e-3 and 3e-4 (T4, fp32), same cells."""
import os, runpy
_base = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_gpt_toggles.py"))["CELLS"]
CELLS = [("md", "# LLM Training Lab, lltGPT step 8 follow-up: post-norm vs pre-norm at lower learning rates (T4, fp32)"),
         ("code", "import os\nos.environ['LLT_JOBS'] = 'lr'")] + _base[1:]
