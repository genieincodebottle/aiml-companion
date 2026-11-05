"""Record the in-page cells' output under real Pyodide 0.27.0 (stdout pasted from run_pyodide.mjs) and CPython.

    node run_pyodide.mjs browser/step1.py > pyo_step1.txt   (and the same for challenge_solution.py)
    python browser/record_pyodide.py pyo_step1.txt pyo_challenge.txt
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from runlog import write_run  # noqa: E402


def cpython(name):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(os.path.join(HERE, name), encoding="utf-8").read(), name, "exec"), {"__name__": "__main__"})
    return buf.getvalue()


def strip(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    return "\n".join(l for l in lines if not l.startswith(("Loading", "Loaded", "[pyodide"))) + "\n"


pyo = {"step1.py": strip(sys.argv[1]), "challenge_solution.py": strip(sys.argv[2])}
cpy = {k: cpython(k) for k in pyo}
same = {k: pyo[k] == cpy[k] for k in pyo}
print(same)
write_run("lltGeneration-t0-pyodide", "T0", "baseline", {"pyodide": "0.27.0 under Node (npm pyodide@0.27.0), numpy preloaded, fresh __main__"}, 0, [],
          evals={"stdoutPyodide": pyo, "stdoutCPython": cpy, "identical": same})
