torchrun --nproc_per_node=1 ddp.py
torchrun --nproc_per_node=2 ddp.py
torchrun --nproc_per_node=2 ddp.py --sync-every-micro
torchrun --nproc_per_node=2 ddp.py --same-seed
