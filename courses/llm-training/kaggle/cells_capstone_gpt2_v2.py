"""Kaggle kernel llt-capstone-gpt2-measure2: the runs v1 missed (40M config, compile at a larger batch,
checkpointing under compile, exact resume), with data read from one parquet shard.
Build: python build_nb.py cells_capstone_gpt2_v2.py llt-capstone-gpt2-measure2 llt-capstone-gpt2-measure2 "LLT Capstone GPT2 Measure2"
"""
import os, runpy

V1 = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_capstone_gpt2.py"))
SETUP = V1["SETUP"].replace("lltCapstoneGPT2_measure.json", "lltCapstoneGPT2_measure_v2.json")
write_cell = V1["write_cell"]

PREP = r'''
out = sh("prep", [sys.executable, "prep_data.py", "/kaggle/working/data", "3000000", "300000"], timeout=420)
RUN["data"] = result_line(out)
save()
'''

PROBE = r'''
env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
r = result_line(sh("probe_40m", [sys.executable, "probe.py", "40m", "0", "8", "12"], timeout=300, env=env1))
RUN["probes"].append(r)
B40 = 12 if r and r["peakGB"].get("12") not in (None, "OOM") and r["peakGB"]["12"] < 12.8 else 8
RUN["picked"] = {"40m": B40}
print(RUN["picked"])
save()
'''


def bench(key, args, budget_min, timeout=600):
    body = ("if time.time() - T_START < %d * 60:\n"
            "    r = result_line(sh(%r, [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', 'bench.py', "
            "'--tag', %r] + %s, timeout=%d))\n"
            "    RUN['benches'].append(r)\n    save()\n" % (budget_min, key, key, args, timeout))
    return ("code", body)


END = r'''
import shutil
shutil.rmtree("/kaggle/working/data", ignore_errors=True)
print("total min", round((time.time() - T_START) / 60, 1))
save()
'''

CELLS = [
    ("md", "# lltCapstoneGPT2 measurements, part 2 (2x T4)\n\n40M config, torch.compile at micro-batch 16, "
           "checkpointing under compile, and an exact-resume check for the 124M model."),
    ("code", SETUP),
    write_cell("gpt.py"), write_cell("prep_data.py"), write_cell("probe.py"), write_cell("bench.py"),
    ("code", PREP),
    ("code", PROBE),
    bench("40m_eager", "['--size', '40m', '--micro', str(B40), '--warmup', '1', '--steps', '4']", 6),
    bench("40m_compile", "['--size', '40m', '--micro', str(B40), '--warmup', '2', '--steps', '4', '--compile']", 7),
    bench("124m_compile_b16", "['--size', '124m', '--micro', '16', '--warmup', '2', '--steps', '3', '--compile']", 9, 420),
    bench("124m_resume", "['--size', '124m', '--micro', '8', '--accum', '2', '--warmup', '1', '--steps', '1', '--resume-check']", 12),
    bench("124m_compile_ckpt_b16", "['--size', '124m', '--micro', '16', '--warmup', '2', '--steps', '2', '--compile', '--ckpt']", 13, 420),
    ("code", END),
]
