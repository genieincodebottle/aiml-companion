del opt
torch.cuda.empty_cache()
params = dict(model.named_parameters())
linear = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head")]
energy, tops = {}, {}
with torch.no_grad():
    for n in linear:
        delta = params[n + ".weight"] - W0[n + ".weight"].to(DEV)          # what full fine-tuning changed
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        energy[n] = ((S ** 2).cumsum(0) / (S ** 2).sum()).cpu()            # energy in the top 1, 2, 3, ... values
        tops[n] = (U[:, :256].cpu(), S[:256].cpu(), Vh[:256].cpu())

print(len(linear), "linear layers. Median share of each update's energy in its top r singular values:")
for r in [1, 4, 16, 64, 256]:
    print(f"  r = {r:>3}  {torch.tensor([float(e[min(r, len(e)) - 1]) for e in energy.values()]).median():.3f}")
need = sorted(int((e < 0.9).sum()) + 1 for e in energy.values())
print("singular values needed for 90% of the energy, median", need[len(need) // 2], "| max", need[-1])

# Rebuild the model from the base plus a rank-r piece of every linear update. Embeddings and norms go back
# to the base, because LoRA on all linear layers cannot change them.
finetuned = {n: p.detach().to("cpu", copy=True) for n, p in params.items()}
for r in [1, 4, 16, 64, 256, "all"]:
    with torch.no_grad():
        for n, p in params.items():
            if n.endswith(".weight") and n[:-7] in tops:
                U, S, Vh = tops[n[:-7]]
                if r == "all":
                    p.copy_(finetuned[n])
                else:
                    p.copy_(W0[n] + (U[:, :r] * S[:r]) @ Vh[:r])
            else:
                p.copy_(W0[n])
    print(f"rank {r!s:>3} slice of every update  held-out loss {mean_loss(model, held, tok.pad_token_id, DEV):.4f}"
          f" | wiki loss {mean_loss(model, wiki, tok.pad_token_id, DEV):.4f}")
