# The long steps have an exit. True (the default, built for a free T4): steps 5, 7, 9 and 12 use our recorded results from the Hub
# instead of running again, and every later step still runs on them. False: you run every step yourself, as we did.
FROM_OUR_CHECKPOINT = True
import json, os, struct
import torch
from huggingface_hub import HfFileSystem, hf_hub_download

torch.set_num_threads(4)
HUB = "aimlcompanion/llm-training-lab"
fs = HfFileSystem()
REC = json.load(open(hf_hub_download(HUB, "lltShip/recorded.json")))     # our A100 results for the long steps, 5 KB

def adapter_base(path):
    # The base model an adapter was trained on, from its adapter_config.json on the Hub
    return json.loads(fs.read_text(HUB + "/" + path + "/adapter_config.json"))["base_model_name_or_path"]

def tensor_shapes(path):
    # Read only the header of a .safetensors file on the Hub: names and shapes, no weights downloaded
    with fs.open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    return {k: v["shape"] for k, v in head.items() if k != "__metadata__"}

print("lltToolUse adapter was trained on", adapter_base("lltToolUse/qwen3.5-2b-lora-s0"))
print("lltDistillation adapter was trained on", adapter_base("lltDistillation/onpolicy-reverse-s0"))
for path in ["lltDPO/sft-s0", "lltDomainAdaptation/cpt-r5-lr3e-5"]:
    cfg = json.loads(fs.read_text(HUB + "/" + path + "/config.json"))
    print(path, "is a full model |", cfg["architectures"][0], "| hidden", cfg["hidden_size"], "| layers", cfg["num_hidden_layers"],
          "| vocab", cfg["vocab_size"])

tool = tensor_shapes(HUB + "/lltToolUse/qwen3.5-2b-lora-s0/adapter_model.safetensors")
qa = [k for k in tool if "q_proj.lora_A" in k][0]
ours = tensor_shapes("Qwen/Qwen3-0.6B-Base/model.safetensors")
print("ToolUse LoRA A for q_proj", tool[qa], "expects inputs of width", tool[qa][1])
print("Qwen3-0.6B-Base q_proj weight", ours["model.layers.0.self_attn.q_proj.weight"], "has inputs of width",
      ours["model.layers.0.self_attn.q_proj.weight"][1])

# Same shapes are not enough. Train a tiny network, then permute its hidden units: the function does not change,
# but the weights now sit in different places, so averaging the two copies breaks it.
g = torch.Generator().manual_seed(0)
x = torch.linspace(-2, 2, 256)[:, None]
y = torch.sin(3 * x) + 0.5 * x

def mlp(w, x):
    return torch.tanh(x @ w["W1"] + w["b1"]) @ w["W2"] + w["b2"]

def loss(w):
    return float(((mlp(w, x) - y) ** 2).mean())

def train(w, steps, lr, seed):
    w = {k: v.clone().requires_grad_(True) for k, v in w.items()}
    opt = torch.optim.Adam(w.values(), lr=lr)
    order = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, 256, (64,), generator=order)
        l = ((mlp(w, x[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad(); l.backward(); opt.step()
    return {k: v.detach() for k, v in w.items()}

init = {"W1": torch.randn(1, 64, generator=g), "b1": torch.randn(64, generator=g) * 0.1,
        "W2": torch.randn(64, 1, generator=g) * 0.1, "b2": torch.zeros(1)}
A = train(init, 3000, 1e-2, seed=1)                       # the "base"
perm = torch.randperm(64, generator=g)
B = {"W1": A["W1"][:, perm], "b1": A["b1"][perm], "W2": A["W2"][perm], "b2": A["b2"]}   # same function, units shuffled
C = train(A, 300, 1e-3, seed=2)                           # a short fine-tune that starts from A
avg = lambda P, Q: {k: (P[k] + Q[k]) / 2 for k in P}
print("loss A", round(loss(A), 5), "| permuted copy B", round(loss(B), 5), "| max output gap A vs B",
      float((mlp(A, x) - mlp(B, x)).abs().max()))
print("average of A and B", round(loss(avg(A, B)), 5), "| average of A and its fine-tune C", round(loss(avg(A, C)), 5))
