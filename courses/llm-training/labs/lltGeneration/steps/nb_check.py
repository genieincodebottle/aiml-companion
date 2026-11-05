import importlib, platform, shutil, sys

# Runtime check. Every line that starts with FIX names the change to make before you go on.
PINS = {"transformers": "5.22.1", "tokenizers": "0.23.3", "huggingface_hub": "2.4.0"}
OURS = {"numpy": "2.5.4"}  # not installed here: torch is built against the runtime's own numpy
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    print("FIX  Python 3.11 or later is needed, and 3.12 is recommended. Colab's default runtime has it.")
for name in ["torch", "numpy", "tokenizers", "huggingface_hub", "transformers"]:
    try:
        v = importlib.import_module(name).__version__
        if name in OURS:
            note = f"   (our CPU runs used {OURS[name]})"
        elif name in PINS and v != PINS[name]:
            note = f"   FIX  run the install cell above, then Runtime > Restart session (pinned {PINS[name]})"
        else:
            note = ""
        print(f"{name:16} {v}{note}")
    except ImportError:
        print(f"FIX  {name} is missing. Run the install cell above.")
free = shutil.disk_usage(".").free / 1e9
print(f"free disk {free:.1f} GB")
if free < 2:
    print("FIX  under 2 GB free. Step 6 downloads SmolLM2-135M (about 270 MB) and the data file is 22 MB.")
try:
    import psutil
    print(f"RAM {psutil.virtual_memory().total / 1e9:.1f} GB")
except ImportError:
    pass
import torch
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none. This lab runs on a CPU, so none is needed.")
