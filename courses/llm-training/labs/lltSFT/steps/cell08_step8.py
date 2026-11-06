import sft_common as C            # written by the notebook's third cell; also labs/lltSFT/sft_common.py in the repo
GPU = torch.device("cuda")
BIG = "HuggingFaceTB/SmolLM2-360M"
tok360 = AutoTokenizer.from_pretrained(BIG)
tok360.chat_template, tok360.eos_token, tok360.pad_token = tok.chat_template, "<|im_end|>", "<|endoftext|>"

def fresh_model():
    return AutoModelForCausalLM.from_pretrained(BIG, dtype=torch.float32).to(GPU)   # fp32 master weights

def make_trainer(model, **changes):
    cfg = dict(output_dir="sft-360m", report_to="none", save_strategy="no",
               fp16=True, bf16=False,          # a T4 has no bf16 tensor cores, and SFTConfig defaults to bf16
               learning_rate=5e-5, lr_scheduler_type="cosine", warmup_steps=10, num_train_epochs=1,
               per_device_train_batch_size=2, gradient_accumulation_steps=8, max_length=1024,
               packing=True, assistant_only_loss=True, logging_steps=5, seed=0, data_seed=0)
    cfg.update(changes)
    data = Dataset.from_list([{"messages": c["messages"]} for c in train])
    return SFTTrainer(model=model, args=SFTConfig(**cfg), processing_class=tok360, train_dataset=data)

model360 = fresh_model()
gen_set = C.generation_set(held)                 # 20 held-out first turns from each split
before = C.eval_loss(model360, tok360, held, GPU, batch_size=1, dtype=torch.float16)["loss"]
g0 = C.eval_generate(model360, tok360, gen_set, GPU, dtype=torch.float16)
print("base | held-out loss", round(before, 4), "| stop rate", g0["stopRate"], "| mean new tokens", g0["meanNewTokens"])
