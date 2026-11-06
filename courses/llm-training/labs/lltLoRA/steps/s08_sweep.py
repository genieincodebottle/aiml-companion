del model
torch.cuda.empty_cache()
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]

# Each run takes about as long as the one in step 8. Add the settings you want to test.
for r, targets in [(1, "all-linear"), (16, ATTENTION)]:
    m = lora_run(r=r, targets=targets)
    del m
    torch.cuda.empty_cache()
