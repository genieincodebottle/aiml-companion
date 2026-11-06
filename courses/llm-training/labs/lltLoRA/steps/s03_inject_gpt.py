from gpt_model import GPT, TINY    # your GPT from the lab Assemble a Modern GPT

def add_lora(model, targets, r=8, alpha=16, rslora=False):
    # Freeze every weight, then wrap each nn.Linear whose name is in targets.
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha, rslora))
    return model

def count(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())

idx = torch.randint(0, TINY.vocab, (2, 64), generator=torch.Generator().manual_seed(0))
for label, targets in [("attention only", {"qkv", "proj"}),
                       ("all linear layers", {"qkv", "proj", "gate", "up", "down"})]:
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    with torch.no_grad():
        before = model(idx)
    add_lora(model, targets, r=8, alpha=16)
    with torch.no_grad():
        after = model(idx)
    trainable, total = count(model)
    print(f"{label:17}  trainable {trainable:>7,} of {total:,} ({100 * trainable / total:.2f}%)  "
          f"logits unchanged: {torch.equal(before, after)}")
print("head stays tied and frozen:", model.head.weight is model.tok.weight, model.head.weight.requires_grad)
