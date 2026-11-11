"""Run lltRL.ipynb top to bottom, headless, and time every cell. Used for the T4 VM re-timing (2026-11-11).

  python run_notebook_timed.py <notebook> <out_dir> [--sub "old=>new"]... [--stop-after N] [--final-dump]
Executes with nbclient in the current directory (cell 10 writes nothing there unless you ask it to). Streams each cell's
stdout to <out_dir>/stdout.txt as it arrives, writes <out_dir>/timing.json (seconds per cell, GPU peak memory the
allocator reports after the GPU cells) and the executed notebook. --sub replaces text in the notebook source first
(for example the START_FROM line); each old string must occur in exactly one cell.
"""
import argparse, json, os, sys, time
import nbformat
from nbclient import NotebookClient

ap = argparse.ArgumentParser()
ap.add_argument("notebook"); ap.add_argument("out")
ap.add_argument("--sub", action="append", default=[])
ap.add_argument("--stop-after", type=int, default=None)
ap.add_argument("--final-dump", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
nb = nbformat.read(a.notebook, as_version=4)
for s in a.sub:
    old, new = s.split("=>", 1)
    hits = [c for c in nb.cells if c.cell_type == "code" and old in c.source]
    assert len(hits) == 1, ("substitution must hit exactly one cell", old, len(hits))
    hits[0].source = hits[0].source.replace(old, new, 1)

PROBE = ('import json as _j\ntry:\n    import torch as _t\n    _p = dict(peakAllocGB=round(_t.cuda.max_memory_allocated(0)/1e9, 3), peakReservedGB=round(_t.cuda.max_memory_reserved(0)/1e9, 3)) if _t.cuda.is_available() else {}\n'
         'except Exception as _e:\n    _p = {}\nprint("PROBE", _j.dumps(_p))')
DUMP = ('import json as _j\n_o = {}\nfor _k, _v in globals().get("runs", {}).items():\n    if isinstance(_v, dict) and "hist" in _v and _v["hist"] and "sec" in _v["hist"][-1]:\n'
        '        _o[_k] = {x: _v.get(x) for x in ("cfg", "firstStep", "sec", "peakGB", "hist", "evals", "groups", "final")}\n'
        'for _n in ("base_eval", "after", "mem"):\n    _v = globals().get(_n)\n    if isinstance(_v, dict):\n        _o[_n] = {k: v for k, v in _v.items() if k != "replies"}\n'
        'if "base_eval" in globals():\n    _o["base_eval_replies"] = base_eval["replies"][:3]\n'
        'if "after" in globals():\n    _o["after_replies"] = after["replies"][:3]\n'
        '_j.dump(_o, open("result.json", "w"), default=str)\nprint("DUMPED", sorted(_o))')

cells, meta = [], []
for i, c in enumerate(nb.cells):
    cells.append(c); meta.append(dict(src=i, probe=False))
    if a.stop_after is not None and i == a.stop_after:
        break
    if c.cell_type == "code":
        cells.append(nbformat.v4.new_code_cell(PROBE)); meta.append(dict(src=i, probe=True))
if a.final_dump:
    cells.append(nbformat.v4.new_code_cell(DUMP)); meta.append(dict(src=-1, probe=True, dump=True))
nb.cells = cells

log = open(os.path.join(a.out, "stdout.txt"), "a", encoding="utf-8")
times, probes = {}, {}


class C(NotebookClient):
    def output(self, outs, msg, display_id, cell_index):
        if msg["msg_type"] == "stream":
            t = msg["content"]["text"]
            log.write(t if t.endswith("\n") else t + "\n"); log.flush()
            if "PROBE " in t:
                probes[cell_index] = json.loads(t.split("PROBE ", 1)[1])
        return super().output(outs, msg, display_id, cell_index)

    async def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        t0 = time.perf_counter()
        log.write("\n=== cell %d (notebook cell %d) start %s\n" % (cell_index, meta[cell_index]["src"], time.strftime("%H:%M:%S"))); log.flush()
        r = await super().async_execute_cell(cell, cell_index, execution_count, store_history)
        times[cell_index] = round(time.perf_counter() - t0, 2)
        log.write("=== cell %d done in %.1f s\n" % (cell_index, times[cell_index])); log.flush()
        return r


t_start = time.time()
cl = C(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": os.path.abspath(a.out)}}, allow_errors=False)
err = None
try:
    cl.execute()
except Exception as e:  # keep the partial timing
    err = repr(e)[:2000]
rows = []
for k, m in enumerate(meta):
    if m["probe"] or k not in times:
        continue
    first = "".join(nb.cells[k].source).strip().splitlines()[0][:70] if nb.cells[k].source.strip() else ""
    rows.append(dict(cell=m["src"], type=nb.cells[k].cell_type, sec=times[k], first=first, peakAfter=probes.get(k + 1)))
code_sec = round(sum(r["sec"] for r in rows if r["type"] == "code"), 1)
json.dump(dict(notebook=os.path.basename(a.notebook), subs=a.sub, error=err, totalCodeSec=code_sec, wallSec=round(time.time() - t_start, 1),
               start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), cells=rows),
          open(os.path.join(a.out, "timing.json"), "w"), indent=1)
nbformat.write(nb, os.path.join(a.out, "executed.ipynb"))
print("DONE code seconds", code_sec, "error", err)
sys.exit(1 if err else 0)
