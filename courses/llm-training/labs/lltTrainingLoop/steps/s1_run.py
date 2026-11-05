chars = prepare()
train_ids = np.memmap("data/train.bin", dtype=np.uint8, mode="r")
val_ids = np.memmap("data/val.bin", dtype=np.uint8, mode="r")
print("vocab", len(chars), "| train tokens", len(train_ids), "| val tokens", len(val_ids))

a = Sampler("data/train.bin", block=64, batch=12, seed=0)
print("windows per epoch", a.n, "| batches per epoch", a.n // a.batch)
for _ in range(5):
    a.next_batch()
saved = a.state_dict()
x_a, y_a = a.next_batch()

b = Sampler("data/train.bin", block=64, batch=12, seed=0)
b.load_state_dict(saved)
x_b, y_b = b.next_batch()
print("saved position", saved, "| same 6th batch after restore", torch.equal(x_a, x_b) and torch.equal(y_a, y_b))
print("targets are inputs shifted by one", torch.equal(x_a[:, 1:], y_a[:, :-1]))
