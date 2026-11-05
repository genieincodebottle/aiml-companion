"""Build a Kaggle kernel folder (notebook + kernel-metadata.json) from a list of cells.

Usage: python build_nb.py <cells.py> <out_dir> <slug> "<Title That Slugifies To The Slug>"
<cells.py> defines CELLS = [("md"|"code", source), ...].
"""
import json, os, runpy, sys

cells_file, out_dir, slug, title = sys.argv[1:5]
cells = runpy.run_path(cells_file)["CELLS"]
nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src}
        if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, slug + ".ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
meta = {
    "id": "genieincodebottle/" + slug, "title": title, "code_file": slug + ".ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    "keywords": [],
}
with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=1)
print("built", out_dir)
