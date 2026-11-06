import gc
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

BASE = "HuggingFaceTB/SmolLM2-360M"
STEPS = int(os.environ.get("LLT_STEPS", "250"))   # the same 250 steps of 16 examples for every arm
TRAIN_LOGS = {}

def sft_tokenizer():
    # The base model ships no chat template. Borrow the instruct model's, and stop on its end-of-turn token.
    tok = AutoTokenizer.from_pretrained(BASE)
    inst = AutoTokenizer.from_pretrained(BASE + "-Instruct")
    tok.chat_template, tok.eos_token = inst.chat_template, inst.eos_token
    tok.pad_token = tok.unk_token
    return tok

def train_arm(rows, name, seed):
    data = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                               "completion": [{"role": "assistant", "content": r["response"]}]} for r in rows])
    tok = sft_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    args = SFTConfig(output_dir="runs/" + name, seed=seed, max_steps=STEPS, per_device_train_batch_size=8,
                     gradient_accumulation_steps=2, learning_rate=5e-5, warmup_steps=10, max_length=1024,
                     completion_only_loss=True, gradient_checkpointing=True, fp16=torch.cuda.is_available(), bf16=False,
                     logging_steps=10, save_strategy="no", report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tok)
    out = trainer.train()
    losses = [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h]
    TRAIN_LOGS[name + "-s" + str(seed)] = {"loss": losses, "seconds": round(out.metrics["train_runtime"], 1)}
    print(name, "seed", seed, "| examples", len(rows), "| epochs", round(STEPS * 16 / len(rows), 2), "| loss", losses[0], "->", losses[-1])
    model = trainer.model.eval()
    del trainer                                     # drops the optimizer state, about 2.9 GB of GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tok
