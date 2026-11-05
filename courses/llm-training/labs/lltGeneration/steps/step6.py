L, H_kv, hd = len(model.blocks), model.blocks[0].h_kv, model.blocks[0].hd
print("layers", L, "| KV heads", H_kv, "| head dim", hd)
for T, dtype, nbytes in [(256, torch.float32, 4), (256, torch.float16, 2)]:
    cache = model.new_cache(1, T, dtype)
    formula = 2 * L * T * H_kv * hd * nbytes
    print(f"T {T}  {str(dtype):13}  formula {formula:,} bytes  tensors {cache.nbytes():,} bytes")
