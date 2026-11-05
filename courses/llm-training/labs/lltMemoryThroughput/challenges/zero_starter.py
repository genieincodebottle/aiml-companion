import numpy as np

def per_gpu_gb(N, n, stage, w=2, g=2, K=12):
    # TODO: training-state GB per GPU for N parameters over n GPUs.
    # Bytes per parameter: w for the weights, g for the gradients, K for the optimizer state.
    # Stage 0 shards nothing. Stage 1 shards the optimizer state over n GPUs,
    # stage 2 also the gradients, stage 3 also the weights.
    return None

# Checks. Run the cell. It prints All checks passed when per_gpu_gb is right.
# ZeRO paper (Rajbhandari et al. 2020): a 7.5B-parameter model on 64 GPUs, fp16 weights and grads, K = 12.
paper = {0: 120.0, 1: 31.4, 2: 16.6, 3: 1.9}
for stage, want in paper.items():
    got = per_gpu_gb(7.5e9, 64, stage)
    assert got is not None, "per_gpu_gb returns None"
    assert abs(got - want) < 0.06, f"stage {stage}: got {got:.2f} GB, the paper reports {want} GB"
# Our FSDP2 run: 1,313,626,112 parameters, fp32 weights and grads (4 bytes each), Adam moments K = 8.
N = 1_313_626_112
one = per_gpu_gb(N, 1, 0, w=4, g=4, K=8)
two = per_gpu_gb(N, 2, 3, w=4, g=4, K=8)
assert abs(one - 21.02) < 0.01, "one T4, no sharding, should need 21.02 GB of training state"
assert abs(two - 10.51) < 0.01, "two T4s, stage 3, should need 10.51 GB each"
for stage in range(4):
    print(f"stage {stage}  7.5B on 64 GPUs  {per_gpu_gb(7.5e9, 64, stage):6.2f} GB per GPU")
print(f"our 1.31B model  one T4 {one:.2f} GB  two T4s at stage 3 {two:.2f} GB each")
print("All checks passed")
