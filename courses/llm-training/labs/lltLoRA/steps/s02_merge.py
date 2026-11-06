torch.manual_seed(1)
with torch.no_grad():
    layer.B.normal_(0, 0.02)                             # stand-in for a trained B, so the update is not zero
w_before = base.weight.detach().clone()
y_adapter = layer(x).detach()

layer.merge()
y_merged = layer(x).detach()
print("merged output matches the adapter output:", torch.allclose(y_merged, y_adapter, atol=1e-5),
      f"| max abs diff {(y_merged - y_adapter).abs().max():.2e}")
delta = base.weight.detach() - w_before
print("rank of the merged update:", int(torch.linalg.matrix_rank(delta)), "of a possible", min(delta.shape))

layer.unmerge()
print("unmerge restores W:", torch.allclose(base.weight, w_before, atol=1e-6),
      f"| max abs diff {(base.weight - w_before).abs().max():.2e}")
