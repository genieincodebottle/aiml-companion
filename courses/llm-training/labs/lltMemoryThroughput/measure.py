# measure.py - rebuild the three Kaggle measurement kernels for this lab.
# Each kernel runs the step files in steps/ exactly as the lab page shows them and records their output.
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KAGGLE = os.path.abspath(os.path.join(HERE, "..", "..", "kaggle"))
for part in ("p1", "p2", "p3"):
    slug = "llt-memthru-" + part
    subprocess.run([sys.executable, os.path.join(KAGGLE, "build_nb.py"), os.path.join(HERE, f"kaggle_cells_{part}.py"),
                    os.path.join(KAGGLE, slug), slug, "LLT Memthru " + part.upper()], check=True)
    print(f"push: kaggle kernels push -p {os.path.join(KAGGLE, slug)} --accelerator NvidiaTeslaT4")
print("after the runs: kaggle kernels output genieincodebottle/<slug> -p kaggle/out/<slug>, then python make_runs.py")
