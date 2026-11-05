python prep.py
CUDA_VISIBLE_DEVICES=0 python sweep.py 10M base 3e-3 1e-2 3e-2 & CUDA_VISIBLE_DEVICES=1 python sweep.py 10M qk 3e-3 1e-2 3e-2 & wait
