# The two broken runs from the page. Each patch breaks one line, runs the step 5 check, then restores it.

def gpt_forward_rope_bug(self, idx, cache=None):
    T = idx.shape[1]
    cos, sin = rope_tables(0, T, self.emb.embedding_dim // self.H)   # BUG, the offset cache.pos is missing
    x = self.emb(idx)
    for i, b in enumerate(self.blocks):
        x = b(x, cos, sin, cache, i)
    if cache is not None:
        cache.pos += T
    return self.head(self.norm(x))

GPT.forward = gpt_forward_rope_bug
print("RoPE offset bug")
check_cache(model, ids, n=60)
GPT.forward = gpt_forward

def block_forward_mask_bug(self, x, cos, sin, cache=None, layer=0):
    B, T, C = x.shape
    q, k, v = self.qkv(self.n1(x)).split([C, self.h_kv * self.hd, self.h_kv * self.hd], dim=2)
    q = q.view(B, T, self.h, self.hd).transpose(1, 2)
    k = k.view(B, T, self.h_kv, self.hd).transpose(1, 2)
    v = v.view(B, T, self.h_kv, self.hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    p = cache.pos if cache is not None else 0
    if cache is not None:
        cache.k[layer, :, :, p:p + T] = k
        cache.v[layer, :, :, p:p + T] = v
        k, v = cache.k[layer, :, :, :p + T], cache.v[layer, :, :, :p + T]
    mask = torch.arange(p + T)[None, :] <= (p + torch.arange(T))[:, None]
    if cache is not None and T > 1:
        mask = None                                                    # BUG, the prefill chunk reads ahead
    k = k.repeat_interleave(self.h // self.h_kv, dim=1)
    v = v.repeat_interleave(self.h // self.h_kv, dim=1)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
    h = self.n2(x)
    return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

Block.forward = block_forward_mask_bug
print("prefill mask bug")
check_cache(model, ids, n=60)
Block.forward = block_forward
