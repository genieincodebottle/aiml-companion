for h_kv in [6, 3, 2, 1]:
    torch.manual_seed(0)
    m = GPT(2048, 192, 4, 6, 256, h_kv).eval()
    cache = m.new_cache(1, 256)
    attn = m.blocks[0].qkv.weight.numel() + m.blocks[0].o.weight.numel()
    print(f"KV heads {h_kv}  group {6 // h_kv}  cache at 256 tokens {cache.nbytes():,} bytes  attention params per layer {attn:,}")
    check_cache(m, ids, n=50)
