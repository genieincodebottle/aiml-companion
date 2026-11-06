trainer = make_trainer(model360)
print(len(trainer.train_dataset), "packed rows of up to 1,024 tokens")
out = trainer.train()
logs = [h for h in trainer.state.log_history if "loss" in h]
print("steps", trainer.state.global_step, "| train loss", round(logs[0]["loss"], 4), "->", round(logs[-1]["loss"], 4),
      "| seconds", round(out.metrics["train_runtime"]), "| peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

model360.eval()
after = C.eval_loss(model360, tok360, held, GPU, batch_size=1, dtype=torch.float16)["loss"]
g1 = C.eval_generate(model360, tok360, gen_set, GPU, dtype=torch.float16)
print("SFT  | held-out loss", round(after, 4), "| stop rate", g1["stopRate"], "| mean new tokens", g1["meanNewTokens"])
print(g1["samples"][20]["text"][:400])
