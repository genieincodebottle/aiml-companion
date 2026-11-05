"""lltIntro timing figures. Run on an idle machine; prints ONE JSON line.

Keys (they fill the [[TIME:key]] placeholders on the lab page):
  t1_notebook_wall_s     wall-clock of lltIntro.ipynb executed end to end on this CPU, model already cached
  t1_notebook_minutes    the same, rounded up to whole minutes (Lab card quickPath.minutes)
  t1_step3_s             wall-clock of step 3 alone (load SmolLM2-135M, random copy, loss, generate)
  t0_step1_cpython_s     wall-clock of the step 1 NumPy code in CPython (stand-in for the browser)
  t0_step1_pyodide_s     wall-clock of step 1 in real Pyodide 0.27.0 under Node, if --pyodide-dir is given
                         (a folder with node_modules/pyodide@0.27.0 and run_pyodide.mjs); else null
Usage: python measure_timing.py [--pyodide-dir DIR]
"""
import argparse
import ast
import contextlib
import io
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, "story-model.json")


def notebook_cells():
    import nbformat
    nb = nbformat.read(os.path.join(HERE, "lltIntro.ipynb"), as_version=4)
    return nb, [c.source for c in nb.cells if c.cell_type == "code"]


def time_notebook():
    import nbformat
    from nbclient import NotebookClient
    work = tempfile.mkdtemp(prefix="lltIntro-timing-")
    shutil.copy(os.path.join(HERE, "lltIntro.ipynb"), work)
    shutil.copy(MODEL_JSON, work)
    nb = nbformat.read(os.path.join(work, "lltIntro.ipynb"), as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": work}}).execute()
    return time.time() - t0


def time_step3():
    _, cells = notebook_cells()
    src = next(c for c in cells if "AutoModelForCausalLM.from_config" in c)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "step3", "exec"), {"__name__": "__main__"})
    return time.time() - t0


def step1_browser_source():
    """Rebuild the page's step 1 code (browser fetch) from the notebook's step 1 cell."""
    _, cells = notebook_cells()
    nb_src = next(c for c in cells if "story-model.json" in c and "def logits" in c)
    body = nb_src[nb_src.index('vocab, H, CTX'):]
    head = ('import base64\nimport numpy as np\nfrom pyodide.http import pyfetch\n\n'
            'M = await (await pyfetch("/data/llt/story-model.json")).json()\n')
    return head + body


def time_step1_cpython():
    class Resp:
        async def json(self):
            return json.load(open(MODEL_JSON, encoding="ascii"))

    async def pyfetch(url):
        return Resp()
    http = types.ModuleType("pyodide.http")
    http.pyfetch = pyfetch
    pkg = types.ModuleType("pyodide")
    pkg.http = http
    sys.modules["pyodide"], sys.modules["pyodide.http"] = pkg, http
    code = compile(step1_browser_source(), "step1", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        res = eval(code, types.ModuleType("__main__").__dict__)
        if asyncio.iscoroutine(res):
            asyncio.run(res)
    return time.time() - t0


def time_step1_pyodide(pdir):
    snippet = os.path.join(tempfile.mkdtemp(prefix="lltIntro-pyo-"), "step1.py")
    open(snippet, "w", encoding="utf-8").write(step1_browser_source())
    out = subprocess.run(["node", "run_pyodide.mjs", snippet, MODEL_JSON], cwd=pdir, capture_output=True, text=True)
    m = re.search(r"run ms (\d+)", out.stdout + out.stderr)
    return int(m.group(1)) / 1000 if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyodide-dir", default=None)
    args = ap.parse_args()
    res = {}
    res["t0_step1_cpython_s"] = round(time_step1_cpython(), 1)
    res["t0_step1_pyodide_s"] = round(time_step1_pyodide(args.pyodide_dir), 1) if args.pyodide_dir else None
    res["t1_step3_s"] = round(time_step3(), 1)
    wall = time_notebook()
    res["t1_notebook_wall_s"] = round(wall, 1)
    res["t1_notebook_minutes"] = math.ceil(wall / 60)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
