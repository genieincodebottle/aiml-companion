"""Kaggle kernel llt-insdata-head (NOT PUSHED; waits for a slot from the coordinator).

The smallest run that executes the rewritten lltInstructionData head cells end to end on the Kaggle image:
cell 1 (runtime check: importlib.metadata, nvidia-smi) and cell 2 (constraints file + %pip install), then an import
check of every package the page cells use and the first lines of the lm-eval IFEval checker. No model download, no
training. Expected: under 5 min of session, about 2 GPU-min of real work (pip install about 90 s on the image).
"""
import os, runpy

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltInstructionData")
head = runpy.run_path(os.path.join(LAB, "notebook_head.py"))

CHECK = r'''
import json, time, platform
from importlib.metadata import version
import torch, numpy, transformers, tokenizers, huggingface_hub, datasets, trl, accelerate, sentence_transformers, langdetect
from lm_eval.tasks.ifeval.utils import process_results
doc = {"key": 1, "prompt": "Write two sentences. Do not use any commas.", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
rec = {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "cuda": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       **{p: version(p) for p in ["transformers", "tokenizers", "huggingface_hub", "datasets", "trl", "accelerate", "sentence-transformers", "lm_eval", "langdetect"]},
       "ifevalCheck": process_results(doc, ["Hello there. It is fine."])["prompt_level_strict_acc"]}
print(json.dumps(rec, indent=1))
json.dump(rec, open("insdata_head_check.json", "w"), indent=1)
'''

CELLS = [("md", "# lltInstructionData head cells on the Kaggle image"), ("code", head["RUNTIME_CHECK"]), ("code", head["INSTALL"]), ("code", CHECK)]
