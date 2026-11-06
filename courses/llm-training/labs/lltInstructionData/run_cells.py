"""Run page cells in order in one namespace, print and record each cell's output.

Usage: python run_cells.py <out.json> cells/c01_pool.py cells/c02_... [...]
Set LLT_SMOKE=1 to make the GPU cells use tiny stand-ins (see each cell); smoke output is not for the page.
"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.__stdout__.reconfigure(encoding="utf-8")
    out_path, cells = os.path.abspath(sys.argv[1]), [os.path.abspath(c) for c in sys.argv[2:]]
    os.chdir(os.environ.get("LLT_WORKDIR", str(HERE)))
    ns = {"__name__": "__main__"}
    subs = {}
    if os.environ.get("LLT_SMOKE") == "1":
        subs = json.load(open(HERE / "dev" / "smoke_subs.json", encoding="utf-8"))
    rec = []
    for c in cells:
        buf = io.StringIO()
        t0 = time.perf_counter()
        print("=" * 20, c, flush=True)
        with contextlib.redirect_stdout(Tee(sys.__stdout__, buf)):
            src = Path(c).read_text(encoding="utf-8")
            for old, new in subs.get(Path(c).name, []):
                assert old in src, (c, old)
                src = src.replace(old, new)
            exec(compile(src, c, "exec"), ns)
        rec.append({"cell": c, "seconds": round(time.perf_counter() - t0, 1), "output": buf.getvalue()})
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
