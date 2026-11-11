# How the T4 measurement was run (2026-11-11)

One Google Cloud VM set up like Kaggle's GPU T4 x2 (n1-highmem-4, 4 vCPU, 26 GB RAM, 2 x T4 15 GB, driver 580, Python 3.12, torch 2.11.0+cu130).
`setup_d.sh` builds the Python environment. `mkvariant.py` copies the learner notebook and changes only the AGG, SCALE, SEED line (and the GPU index for the second T4).
`run_cell.sh` runs that notebook top to bottom with `jupyter nbconvert --execute` (repeating the pass if the 100-minute session limit stops training before `done.json`).
`chain.sh` starts the next cell on a GPU when the previous one finishes. Two cells ran at once, one per T4. `make_runs_t4.py` turns the pulled folders into the run files.
