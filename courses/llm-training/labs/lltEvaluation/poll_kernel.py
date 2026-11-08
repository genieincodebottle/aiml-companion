"""Wait until a Kaggle kernel leaves RUNNING/QUEUED, then download its output (page through every file).
Usage: python poll_kernel.py <slug> <out dir> [poll seconds]
Needs KAGGLE_API_TOKEN in the environment.
"""
import os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi

slug, out = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 120
api = KaggleApi()
api.authenticate()
t0 = time.time()
while True:
    try:
        st = str(api.kernels_status("genieincodebottle/" + slug))
    except Exception as e:                      # transient network errors: wait and ask again
        print("status check failed:", type(e).__name__, flush=True)
        time.sleep(every)
        continue
    if "RUNNING" not in st and "QUEUED" not in st:
        break
    time.sleep(every)
print("status", st, "after", round((time.time() - t0) / 60, 1), "min", flush=True)
os.makedirs(out, exist_ok=True)
api.kernels_output("genieincodebottle/" + slug, path=out, force=True, quiet=True)
print("downloaded to", out, flush=True)
