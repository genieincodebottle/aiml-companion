"""Wait until a Kaggle kernel leaves RUNNING/QUEUED, then download its output files that match a pattern.
Usage: python poll_kernel.py <slug> <out dir> [file regex] [poll seconds]
Needs KAGGLE_API_TOKEN in the environment (from ~/.kaggle/access_token; never printed).
"""
import os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi

slug, out = sys.argv[1], sys.argv[2]
pattern = sys.argv[3] if len(sys.argv) > 3 else None
every = int(sys.argv[4]) if len(sys.argv) > 4 else 300
api = KaggleApi()
api.authenticate()
t0 = time.time()
while True:
    st = str(api.kernels_status("genieincodebottle/" + slug))
    if "RUNNING" not in st and "QUEUED" not in st:
        break
    time.sleep(every)
print("status", st, "after", round((time.time() - t0) / 60, 1), "min", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)
os.makedirs(out, exist_ok=True)
files, _ = api.kernels_output("genieincodebottle/" + slug, path=out, file_pattern=pattern, force=True, quiet=True)
print("downloaded", len(files or []), "files to", out, flush=True)
