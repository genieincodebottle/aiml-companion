import shutil, subprocess, sys
from safetensors import safe_open

CFG = f"""merge_method: ties
base_model: {DIRS["base"]}
models:
  - model: {DIRS["sft"]}
    parameters: {{weight: 1.0, density: 0.2}}
  - model: {DIRS["hindi"]}
    parameters: {{weight: 1.0, density: 0.2}}
parameters: {{normalize: true, lambda: 1.0}}
dtype: float32
out_dtype: float32
"""
open("ties.yml", "w").write(CFG)
t0 = time.time()
subprocess.run(["mergekit-yaml", "ties.yml", "merged-mergekit", "--quiet", "--random-seed", "0"], check=True)
print("mergekit took", round(time.time() - t0), "s")

t0 = time.time()
ours = merge_model("ties", density=0.2, dtype=torch.float32)
print("your TIES took", round(time.time() - t0), "s")
worst, same, n = 0.0, 0, 0
for f in sorted(glob.glob("merged-mergekit/*.safetensors")):
    with safe_open(f, "pt") as g:
        for k in g.keys():
            gap = float((g.get_tensor(k) - ours[k]).abs().max())
            worst, same, n = max(worst, gap), same + (gap == 0), n + 1
print(n, "tensors compared |", same, "identical | largest gap", worst)
del ours
shutil.rmtree("merged-mergekit")          # 2.4 GB of fp32 weights you no longer need
