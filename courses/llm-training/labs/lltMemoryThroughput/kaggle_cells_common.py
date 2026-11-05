# Shared pieces for the lltMemoryThroughput Kaggle measurement kernels.
# Every step file is run with exec() exactly as shipped on the lab page, and its stdout is recorded.
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = os.path.join(HERE, "steps")


def src(name):
    with open(os.path.join(STEPS, name), encoding="utf-8") as f:
        return f.read()


SETUP = r'''
import io, json, os, sys, time, contextlib, platform, subprocess
RUN = {"module": "lltMemoryThroughput", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stdout": {}, "data": {}}

class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)

def run_step(key, code):
    buf = Tee()
    t = time.time()
    with contextlib.redirect_stdout(buf):
        exec(code, globals())
    RUN["stdout"][key] = buf.getvalue()
    RUN.setdefault("stepSec", {})[key] = round(time.time() - t, 1)
    print(buf.getvalue())

def save(name):
    import torch
    RUN["env"] = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    RUN["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/kaggle/working/{name}.json", "w") as f:
        json.dump(RUN, f, indent=1, default=str)
    print("saved", name)
'''


def write_file_cell(fname, text):
    return "with open(" + repr(fname) + ", 'w') as f:\n    f.write(" + repr(text) + ")\nprint('wrote', " + repr(fname) + ")\n"


def step_cell(key, fname):
    return "run_step(" + repr(key) + ", " + repr(src(fname)) + ")\n"
