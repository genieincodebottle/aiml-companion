"""Write lltTinyStories.ipynb from cells_lab.py. Usage: python build_lab_nb.py"""
import json, os, runpy
here = os.path.dirname(os.path.abspath(__file__))
cells = runpy.run_path(os.path.join(here, "cells_lab.py"))["CELLS"]
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s.strip()} if k == "md" else
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()}
                for k, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(here, "lltTinyStories.ipynb"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote lltTinyStories.ipynb with", len(cells), "cells")
