import torch

@torch.no_grad()
def extraction(model, canaries, eot, prefix=50, suffix=50):
    """canaries: LongTensor [n, prefix + suffix]. Greedy-decode suffix tokens after EOT + the
    first prefix tokens. A canary is extracted when every generated token matches."""
    model.eval()
    start = torch.full((canaries.size(0), 1), eot, dtype=torch.long, device=canaries.device)
    x = torch.cat([start, canaries[:, :prefix]], dim=1)
    for _ in range(suffix):
        next_id = model(x)[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)
    match = x[:, 1 + prefix:] == canaries[:, prefix:prefix + suffix]
    return match.all(dim=1).float().mean().item(), match.float().mean().item()

# A check you can run on a CPU. A "model" that has stored the canaries extracts all of them.
torch.manual_seed(0)
V, EOT = 4096, 0
canaries = torch.randint(1, V, (4, 100))
table = {}
for row in canaries.tolist():
    for i in range(1, 100):
        table[tuple(row[:i])] = row[i]

def lookup_model(x):
    out = torch.zeros(x.size(0), x.size(1), V)
    for r, seq in enumerate(x.tolist()):
        out[r, -1, table.get(tuple(seq[1:]), 1)] = 1.0
    return out

lookup_model.eval = lambda: None
print("extraction rate, token match:", extraction(lookup_model, canaries, EOT))
