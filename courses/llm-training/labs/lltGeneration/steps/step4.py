class KVCache:
    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, dtype=torch.float32):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype)   # allocated once, filled in place
        self.v = torch.zeros(shape, dtype=dtype)
        self.pos = 0                               # slots that hold real tokens
    def nbytes(self):
        return self.k.nbytes + self.v.nbytes

def block_forward(self, x, cos, sin, cache=None, layer=0):
    B, T, C = x.shape
    q, k, v = self.qkv(self.n1(x)).split([C, self.h_kv * self.hd, self.h_kv * self.hd], dim=2)
    q = q.view(B, T, self.h, self.hd).transpose(1, 2)
    k = k.view(B, T, self.h_kv, self.hd).transpose(1, 2)
    v = v.view(B, T, self.h_kv, self.hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    p = cache.pos if cache is not None else 0
    if cache is not None:
        cache.k[layer, :, :, p:p + T] = k          # write the new keys and values
        cache.v[layer, :, :, p:p + T] = v
        k, v = cache.k[layer, :, :, :p + T], cache.v[layer, :, :, :p + T]
    mask = torch.arange(p + T)[None, :] <= (p + torch.arange(T))[:, None]   # query i reads keys 0 .. p + i
    k = k.repeat_interleave(self.h // self.h_kv, dim=1)
    v = v.repeat_interleave(self.h // self.h_kv, dim=1)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
    h = self.n2(x)
    return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

def gpt_forward(self, idx, cache=None):
    T = idx.shape[1]
    start = cache.pos if cache is not None else 0   # the RoPE position offset
    cos, sin = rope_tables(start, T, self.emb.embedding_dim // self.H)
    x = self.emb(idx)
    for i, b in enumerate(self.blocks):
        x = b(x, cos, sin, cache, i)
    if cache is not None:
        cache.pos += T
    return self.head(self.norm(x))

def new_cache(self, batch, max_len, dtype=torch.float32):
    b = self.blocks[0]
    return KVCache(len(self.blocks), batch, b.h_kv, max_len, b.hd, dtype)

Block.forward, GPT.forward, GPT.new_cache = block_forward, gpt_forward, new_cache
cache = model.new_cache(1, 64)
print(model(torch.tensor([ids]), cache).shape, "cache.pos", cache.pos)
