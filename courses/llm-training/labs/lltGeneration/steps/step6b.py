from transformers import AutoModelForCausalLM

smol = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32).eval()
c = smol.config
T = 512
x = torch.randint(0, c.vocab_size, (1, T), generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    pkv = smol(x, use_cache=True).past_key_values
real = sum(layer.keys.nbytes + layer.values.nbytes for layer in pkv.layers)
hd = c.hidden_size // c.num_attention_heads
formula = 2 * c.num_hidden_layers * T * c.num_key_value_heads * hd * 4
print("layers", c.num_hidden_layers, "| KV heads", c.num_key_value_heads, "| head dim", hd)
print(f"formula {formula:,} bytes  cache tensors {real:,} bytes")
