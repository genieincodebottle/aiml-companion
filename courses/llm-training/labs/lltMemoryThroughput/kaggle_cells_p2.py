import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cells_common import SETUP, src, write_file_cell

RUNCMD = r'''
def torchrun(key, args, timeout=1800):
    cmd = [sys.executable, "-m", "torch.distributed.run"] + args
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + ("" if p.returncode == 0 else "\nSTDERR TAIL\n" + p.stderr[-4000:])
    RUN["stdout"][key] = out
    RUN.setdefault("stepSec", {})[key] = round(time.time() - t, 1)
    RUN.setdefault("returncode", {})[key] = p.returncode
    print(key, "rc", p.returncode, round(time.time() - t, 1), "s")
    print(out)
'''

CELLS = [
    ("md", "# lltMemoryThroughput part 2 measurements (2x T4)\n\nRuns allreduce.py, ddp.py, fsdp.py and fsdp_ckpt.py exactly as shipped with torchrun and records their output."),
    ("code", SETUP + RUNCMD + "\nimport torch\nprint(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\nprint(subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True).stdout)\nRUN['data']['topo'] = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True).stdout\n"),
    ("code", write_file_cell("gpt.py", src("gpt.py"))),
    ("code", write_file_cell("allreduce.py", src("allreduce.py"))),
    ("code", write_file_cell("ddp.py", src("ddp.py"))),
    ("code", write_file_cell("fsdp.py", src("fsdp.py"))),
    ("code", write_file_cell("fsdp_ckpt.py", src("fsdp_ckpt.py"))),
    ("code", "torchrun('allreduce', ['--nproc_per_node=2', 'allreduce.py'])\nsave('lltMemoryThroughput_p2')\n"),
    ("code", "torchrun('ddp_1gpu', ['--nproc_per_node=1', 'ddp.py'])\n"
             "torchrun('ddp_2gpu', ['--nproc_per_node=2', 'ddp.py'])\n"
             "torchrun('ddp_2gpu_sync_every_micro', ['--nproc_per_node=2', 'ddp.py', '--sync-every-micro'])\n"
             "torchrun('ddp_2gpu_same_seed', ['--nproc_per_node=2', 'ddp.py', '--same-seed'])\nsave('lltMemoryThroughput_p2')\n"),
    ("code", "torchrun('fsdp_1gpu', ['--nproc_per_node=1', 'fsdp.py'])\n"
             "torchrun('fsdp_2gpu', ['--nproc_per_node=2', 'fsdp.py'])\nsave('lltMemoryThroughput_p2')\n"),
    ("code", "torchrun('fsdp_ckpt', ['--nproc_per_node=2', 'fsdp_ckpt.py'])\n"
             "import shutil\nshutil.rmtree('/tmp/ckpt_fsdp', ignore_errors=True)\nsave('lltMemoryThroughput_p2')\n"),
]
