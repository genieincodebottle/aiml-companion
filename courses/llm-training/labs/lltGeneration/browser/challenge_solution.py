import numpy as np

def kv_bytes(layers, tokens, kv_heads, head_dim, bytes_per, batch=1):
    # 2 tensors (K and V) per layer, each batch x kv_heads x tokens x head_dim elements
    return 2 * layers * batch * kv_heads * tokens * head_dim * bytes_per

def max_tokens(budget_bytes, layers, kv_heads, head_dim, bytes_per, batch=1):
    per_token = kv_bytes(layers, 1, kv_heads, head_dim, bytes_per, batch)
    return budget_bytes // per_token

# Checks. Run the cell. It prints All checks passed when both functions are right.
cache = np.zeros((4, 2, 1, 6, 256, 32), dtype=np.float32)  # layers, K and V, batch, KV heads, tokens, head dim
assert kv_bytes(4, 256, 6, 32, 4) == cache.nbytes, "the lab model's float32 cache at 256 tokens is " + str(cache.nbytes) + " bytes"
assert kv_bytes(30, 1, 3, 64, 2) == 23_040, "SmolLM2-135M in bf16 stores 23,040 bytes per token"
qwen = kv_bytes(28, 4096, 8, 128, 2)
assert qwen == 469_762_048, "Qwen3-0.6B at 4,096 tokens in bf16 should be 469,762,048 bytes"
assert kv_bytes(28, 4096, 8, 128, 2, batch=8) == 8 * qwen, "the cache grows linearly with the batch"
assert kv_bytes(28, 4096, 16, 128, 2) == 2 * qwen, "16 KV heads instead of 8 doubles the cache"
n = max_tokens(2**30, 28, 8, 128, 2)
assert n == 9_362, "1 GiB holds 9,362 tokens of Qwen3-0.6B cache in bf16"
assert kv_bytes(28, n + 1, 8, 128, 2) > 2**30, "one more token should no longer fit"
print("Qwen3-0.6B, 4,096 tokens:", qwen, "bytes |", n, "tokens fit in 1 GiB")
print("All checks passed")
