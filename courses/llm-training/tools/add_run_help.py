"""Insert (or refresh) the standard "How to run this notebook" cell in every LLM Training Lab notebook.

Owner rule 2026-11-09: no beginner friction around Kaggle, Colab and Hugging Face. Every learner notebook gets the
same short cell right after its title cell. Idempotent: the cell carries a marker and is replaced in place.
The per-module build_notebook.py scripts do not know about this cell, so run this again after any rebuild:

    python courses/llm-training/tools/add_run_help.py

A notebook is rewritten only if re-serialising it unchanged reproduces the file byte for byte (so nothing else moves).
"""
import json
import pathlib
import sys

MARK = "<!-- llt-run-help -->"
TEXT = [
    MARK + "\n",
    "## How to run this notebook\n",
    "\n",
    "- **Check the lab card** on the module page first. It says whether this lab needs a GPU and how long it took us.\n",
    "- **Kaggle** (where we recorded most runs): use the lab page's *Open on Kaggle* button. If the lab needs a GPU, "
    "open the notebook settings and pick *Accelerator: GPU T4 x2* (not P100). Switch *Internet* on so the install cell "
    "and the downloads work. Kaggle asks you to verify a phone number once before it allows a GPU or internet.\n",
    "- **Colab**: use the lab page's *Open in Colab* button. If the lab needs a GPU, pick "
    "*Runtime > Change runtime type > T4 GPU*.\n",
    "- **Hugging Face**: no account or token is needed. Our checkpoints and data files download from the public repo "
    "[aimlcompanion/llm-training-lab](https://huggingface.co/aimlcompanion/llm-training-lab).\n",
    "- **Run the cells in order.** Code cell numbers match the lab page. Your numbers should land close to ours, "
    "not match exactly: GPU runs are not bit-for-bit repeatable.",
]

root = pathlib.Path(__file__).resolve().parents[1] / "labs"
changed, skipped = [], []
for nb_path in sorted(root.glob("llt*/llt*.ipynb")):
    raw = nb_path.read_bytes().decode("utf-8")
    nb = json.loads(raw)
    fmt = None
    for ensure_ascii in (False, True):
        for tail in ("\n", ""):
            if json.dumps(nb, indent=1, ensure_ascii=ensure_ascii) + tail == raw.replace("\r\n", "\n"):
                fmt = (ensure_ascii, tail)
    if fmt is None:
        skipped.append(nb_path.name)
        continue
    cell = {"cell_type": "markdown", "metadata": {}, "source": TEXT}
    cells = nb["cells"]
    idx = next((i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and MARK in "".join(c["source"])), None)
    if idx is None:
        cells.insert(1, cell)
    elif cells[idx]["source"] == TEXT:
        continue
    else:
        cells[idx] = cell
    out = json.dumps(nb, indent=1, ensure_ascii=fmt[0]) + fmt[1]
    if "\r\n" in raw:
        out = out.replace("\n", "\r\n")
    nb_path.write_bytes(out.encode("utf-8"))
    changed.append(nb_path.name)

print("changed:", len(changed), changed)
print("skipped (format not reproducible, left alone):", skipped)
sys.exit(1 if skipped else 0)
