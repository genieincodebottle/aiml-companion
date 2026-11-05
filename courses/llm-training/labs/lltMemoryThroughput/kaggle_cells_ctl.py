import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cells_common import SETUP, src, write_file_cell

HERE = os.path.dirname(os.path.abspath(__file__))
CTL = open(os.path.join(HERE, "diag", "fsdp_ckpt_control.py"), encoding="utf-8").read()

RUNCELL = r'''
t = time.time()
p = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', 'fsdp_ckpt_control.py'], capture_output=True, text=True)
RUN['stdout']['control'] = p.stdout + ('' if p.returncode == 0 else p.stderr[-4000:])
RUN['stepSec'] = {'control': round(time.time() - t, 1)}
print(RUN['stdout']['control'])
import shutil
shutil.rmtree('/tmp/ckpt_control', ignore_errors=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'tokenizers'], check=False)
import importlib
RUN['data']['versions'] = {m: importlib.import_module(m).__version__ for m in ('datasets', 'tokenizers', 'numpy')}
RUN['data']['ramGB'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1)
print(RUN['data'])
save('lltMemoryThroughput_ctl')
'''

CELLS = [
    ("md", "# lltMemoryThroughput checkpoint negative control (2x T4)\n\nReload with and without optimizer state, and checksum every saved tensor."),
    ("code", SETUP),
    ("code", write_file_cell("gpt.py", src("gpt.py")) + write_file_cell("fsdp.py", src("fsdp.py")) + write_file_cell("fsdp_ckpt_control.py", CTL)),
    ("code", RUNCELL),
]
