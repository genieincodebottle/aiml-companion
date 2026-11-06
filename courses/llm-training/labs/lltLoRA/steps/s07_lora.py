import os
from peft import LoraConfig, get_peft_model

del model, params, finetuned, W0, tops
torch.cuda.empty_cache()

def lora_run(r=16, targets="all-linear", lr=LORA_LR, seed=0, save=None):
    model = load()
    model.enable_input_require_grads()                   # gradient checkpointing needs a grad on the inputs
    torch.manual_seed(seed)                              # sets the random init of every A
    cfg = LoraConfig(r=r, lora_alpha=32, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"r = {r}, {sum(p.numel() for p in trainable):,} trainable parameters")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    hist = train(model, train_rows, 125, opt, tok.pad_token_id, DEV, budget=LORA_BUDGET, seed=seed)
    report(f"LoRA r{r} " + (targets if isinstance(targets, str) else "attention"), model, hist)
    if save:
        model.save_pretrained(save)                      # writes the adapter only
        print(save, sum(os.path.getsize(os.path.join(save, f)) for f in os.listdir(save)) / 1e6, "MB")
    return model

model = lora_run(save="lora-r16")
