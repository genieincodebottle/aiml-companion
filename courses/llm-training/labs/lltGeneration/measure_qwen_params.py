"""Count Qwen3-0.6B's distinct parameters (tied embeddings counted once) and its bf16 weight bytes.

Used by interview answer 2 (KV cache vs weights). Writes runs/lltGeneration/lltGeneration-t1-qwen3-params.json.

    ../../.venv-latest/Scripts/python.exe measure_qwen_params.py
"""
import json

import torch
from transformers import AutoModelForCausalLM

from runlog import write_run

torch.set_num_threads(2)
name = "Qwen/Qwen3-0.6B"
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
c = m.config
distinct = sum(p.numel() for p in m.parameters())               # parameters() yields a tied tensor once
emb = m.get_input_embeddings().weight
tied = m.get_output_embeddings().weight.data_ptr() == emb.data_ptr()
bytes_bf16 = sum(p.numel() * p.element_size() for p in m.parameters())
kv_per_token = 2 * c.num_hidden_layers * c.num_key_value_heads * c.head_dim * 2
kv_8x4096 = kv_per_token * 4096 * 8
ev = {"model": name, "distinctParams": distinct, "embeddingParams": emb.numel(), "tieWordEmbeddings": bool(c.tie_word_embeddings),
      "outputSharesInputEmbedding": tied, "weightBytesBf16": bytes_bf16, "kvBytesPerTokenBf16": kv_per_token,
      "kvBytes8Users4096": kv_8x4096, "kvOverWeights": round(kv_8x4096 / bytes_bf16, 2),
      "layers": c.num_hidden_layers, "kvHeads": c.num_key_value_heads, "headDim": c.head_dim, "queryHeads": c.num_attention_heads}
print(json.dumps(ev, indent=1))
write_run("lltGeneration-t1-qwen3-params", "T1", "baseline", {"model": name, "dtype": "bfloat16"}, 0, [], evals=ev)
