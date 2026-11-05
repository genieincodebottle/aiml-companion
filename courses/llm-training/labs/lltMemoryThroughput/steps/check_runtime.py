# Runtime check. Run this first. It prints the menu path that fixes each problem it finds.
import os, shutil, sys
import torch

problems = []
if sys.version_info < (3, 11):
    problems.append(f"Python {sys.version.split()[0]} is too old. Kaggle and Colab images ship 3.11 or later.")
if not torch.cuda.is_available():
    problems.append("No GPU. Kaggle: Settings > Accelerator > GPU T4 x2. Colab: Runtime > Change runtime type > T4 GPU.")
else:
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("GPUs:", names, "| capability", torch.cuda.get_device_capability(0))
    if any("P100" in n for n in names):
        problems.append("P100 found. Current PyTorch wheels dropped it. Kaggle: Settings > Accelerator > GPU T4 x2.")
    if len(names) < 2:
        print("One GPU. Part 2 (two GPUs) needs Kaggle: Settings > Accelerator > GPU T4 x2. Parts 1 and 3 run here.")
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
disk = shutil.disk_usage("/tmp").free / 1e9
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | RAM {ram:.0f} GB | free disk in /tmp {disk:.0f} GB")
if disk < 20:
    problems.append("Under 20 GB free in /tmp. Step 10 writes a 16.2 GB checkpoint there.")
for p in problems or ["Runtime OK"]:
    print(p)
