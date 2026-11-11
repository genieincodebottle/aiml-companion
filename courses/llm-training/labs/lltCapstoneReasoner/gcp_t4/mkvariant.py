import json, sys
src, dst, gpu, agg, scale, seed = sys.argv[1:7]
nb = json.load(open(src))
for c in nb["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "code" and "CUDA_VISIBLE_DEVICES" in s:
        s = s.replace("os.environ['CUDA_VISIBLE_DEVICES'] = '0'", "os.environ['CUDA_VISIBLE_DEVICES'] = '%s'" % gpu)
    if c["cell_type"] == "code" and s.startswith("AGG, SCALE, SEED"):
        s = s.replace("AGG, SCALE, SEED = 'grpo', 1, 0", "AGG, SCALE, SEED = '%s', %s, %s" % (agg, scale, seed))
    c["source"] = s.splitlines(True)
json.dump(nb, open(dst, "w"), indent=1)
