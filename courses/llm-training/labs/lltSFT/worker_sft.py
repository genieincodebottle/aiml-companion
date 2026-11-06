"""GPU worker for the lltSFT Run Library: full SFT of SmolLM2-360M with TRL on one T4, one run after another.

Usage: python worker_sft.py <queue.json> <data_cache.json> <out_dir>
CUDA_VISIBLE_DEVICES must name one GPU (SFTTrainer refuses nn.DataParallel). Each run writes <out_dir>/<runId>.json
as soon as it ends, so a session that dies keeps the finished runs.

Run config keys: runId, label, seed, model, assistantOnly (bool), packing ("bfd" | "naive" | "none"), epochs,
padIsEos (bool, the broken run), evalEachEpoch (bool), genAtStart (bool), rawPromptGen (bool), lossCheck (bool),
evalOnly (bool, for the instruct reference).
"""
import gc
import json
import os
import platform
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sft_common as C  # noqa: E402

queue_path, data_path, out_dir = sys.argv[1:4]
QUEUE = json.load(open(queue_path))
DATA = json.load(open(data_path, encoding="utf-8"))
TRAIN, HELD = DATA["train"], DATA["eval"]
GEN = C.generation_set(HELD)
DEV = torch.device("cuda")
AMP = torch.float16   # T4: capability (7, 5), no bf16 tensor cores
DRY = os.environ.get("LLT_DRYRUN") == "1"   # CPU smoke test of this script: tiny data, SmolLM2-135M, float32
if DRY:
    DEV, AMP = torch.device("cpu"), None
    TRAIN, HELD = TRAIN[:24], HELD[:6]
    GEN = HELD[:2]
    torch.set_num_threads(4)
GEN_TOKENS = 16 if DRY else 512

import transformers  # noqa: E402
import trl  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

ENV = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "transformers": transformers.__version__, "trl": trl.__version__,
       "datasets": __import__("datasets").__version__, "accelerate": __import__("accelerate").__version__,
       "gpu": torch.cuda.get_device_name(0) if not DRY else "cpu",
       "capability": list(torch.cuda.get_device_capability(0)) if not DRY else None}
print("env", ENV, flush=True)


def naive_collator(collate):
    """Same packed rows and labels as TRL's collator, but positions never reset, so documents attend to each other."""
    def wrapped(examples):
        b = collate(examples)
        b["position_ids"] = torch.arange(b["input_ids"].shape[1])[None]
        for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx"):
            b.pop(k, None)
        return b
    return wrapped


def pad_is_eos_collator(collate, pad_id):
    """The classic bug: pad = EOS, then every label equal to the pad id is set to -100 (DataCollatorForLanguageModeling)."""
    def wrapped(examples):
        b = collate(examples)
        b["labels"][b["labels"] == pad_id] = -100
        return b
    return wrapped


def sync():
    if not DRY:
        torch.cuda.synchronize()


class Recorder(TrainerCallback):
    def __init__(self, run, tok):
        self.run, self.tok, self.evals = run, tok, []

    def on_train_begin(self, args, state, control, model=None, **kw):
        sync()
        self.t = time.time()

    def on_epoch_end(self, args, state, control, model=None, **kw):
        if self.run.get("evalEachEpoch"):
            sync()
            t = time.time()
            e = C.eval_loss(model, self.tok, HELD, DEV, batch_size=1, dtype=AMP)
            self.evals.append({"epoch": round(state.epoch, 3), "step": state.global_step, "evalLoss": round(e["loss"], 5)})
            print(self.run["runId"], "epoch eval", self.evals[-1], flush=True)
            self.t += time.time() - t   # eval time is not training time

    def on_train_end(self, args, state, control, **kw):
        sync()
        self.sec = time.time() - self.t


def one(run):
    t_run = time.time()
    if not DRY:
        torch.cuda.reset_peak_memory_stats()
    rec = {"runId": run["runId"], "module": "lltSFT", "tier": "T2", "label": run["label"],
           "hardware": {"gpu": ENV["gpu"], "count": 1, "vramGB": None if DRY else round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)},
           "env": ENV, "config": run, "seed": run.get("seed")}
    model_id = run.get("model", C.BASE_360)
    if DRY:
        model_id = C.INSTRUCT_135 if run.get("evalOnly") else C.BASE_135
    if run.get("evalOnly"):
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.pad_token = "<|endoftext|>"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEV)
        rec["evals"] = {"evalLoss": C.eval_loss(model, tok, HELD, DEV, batch_size=1, dtype=AMP)["loss"]}
        g = C.eval_generate(model, tok, GEN, DEV, dtype=AMP, max_new_tokens=GEN_TOKENS)
        rec["evals"]["generation"] = {k: v for k, v in g.items() if k != "samples"}
        rec["samples"] = g["samples"][:60]
        rec["wallClockSec"] = round(time.time() - t_run, 1)
        return rec
    tok = C.setup_tokenizer(AutoTokenizer.from_pretrained(model_id), AutoTokenizer.from_pretrained(C.INSTRUCT_360))
    if run.get("padIsEos"):
        tok.pad_token = tok.eos_token          # the instruct tokenizer ships exactly this: pad = <|im_end|>
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEV)
    evals = {}
    if run.get("genAtStart"):
        evals["base"] = {"evalLoss": C.eval_loss(model, tok, HELD, DEV, batch_size=1, dtype=AMP)["loss"]}
        g = C.eval_generate(model, tok, GEN, DEV, dtype=AMP, max_new_tokens=GEN_TOKENS)
        evals["base"]["generation"] = {k: v for k, v in g.items() if k != "samples"}
        rec["baseSamples"] = g["samples"][:6]
    packing = run["packing"] != "none"
    bs = 2 if packing else 4
    args = SFTConfig(
        output_dir="/tmp/sft-" + run["runId"], report_to="none", save_strategy="no", eval_strategy="no",
        fp16=not DRY, bf16=False, use_cpu=DRY, seed=run["seed"], data_seed=run["seed"],
        learning_rate=5e-5, lr_scheduler_type="cosine", warmup_steps=10, weight_decay=0.0,
        num_train_epochs=run.get("epochs", 1), per_device_train_batch_size=bs, gradient_accumulation_steps=8,
        max_length=C.MAX_TOKENS, packing=packing, packing_strategy="bfd",
        assistant_only_loss=run["assistantOnly"], logging_steps=5, gradient_checkpointing=True,
        dataloader_num_workers=0,
    )
    rcb = Recorder(run, tok)
    tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list([{"messages": c["messages"]} for c in TRAIN]),
                    processing_class=tok, callbacks=[rcb])
    if run["packing"] == "naive":
        tr.data_collator = naive_collator(tr.data_collator)
    if run.get("padIsEos"):
        tr.data_collator = pad_is_eos_collator(tr.data_collator, tok.pad_token_id)
    rec["trainRows"] = len(tr.train_dataset)
    if run.get("lossCheck"):
        # Build it, then prove it: our masked NLL equals TRL's compute_loss on the same packed batch.
        b = next(iter(tr.get_train_dataloader()))
        b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        model.eval()
        with torch.no_grad():
            lib = tr.compute_loss(tr.model, dict(b)).item()
            logits = model(input_ids=b["input_ids"], position_ids=b["position_ids"], use_cache=False).logits.double()
            tgt = b["labels"][:, 1:]
            m = tgt != -100
            own = torch.nn.functional.cross_entropy(logits[:, :-1][m], tgt[m]).item()
        model.train()
        rec["lossCheck"] = {"own": own, "library": lib, "absDiff": abs(own - lib), "dtype": "model float32, own loss in float64",
                            "batchShape": list(b["input_ids"].shape), "lossTokens": int(m.sum()),
                            "docsInBatch": int((b["position_ids"] == 0).sum())}
        print(run["runId"], "lossCheck", rec["lossCheck"], flush=True)
        del logits
    if run.get("evalEachEpoch"):
        rcb.evals.append({"epoch": 0, "step": 0, "evalLoss": round(C.eval_loss(model, tok, HELD, DEV, batch_size=1, dtype=AMP)["loss"], 5)})
    tr.train()
    logs = [h for h in tr.state.log_history if "loss" in h]
    rec["metrics"] = [{"step": h["step"], "trainLoss": h["loss"], "gradNorm": h.get("grad_norm"), "lr": h.get("learning_rate"),
                       "tokens": h.get("num_tokens"), "tokenAccuracy": h.get("mean_token_accuracy"), "entropy": h.get("entropy")}
                      for h in logs]
    tokens = logs[-1].get("num_tokens") if logs else None
    rec["train"] = {"steps": tr.state.global_step, "trainSec": round(rcb.sec, 1),
                    "secPerStep": round(rcb.sec / max(1, tr.state.global_step), 3),
                    "tokensSeen": tokens, "tokensPerSec": round(tokens / rcb.sec, 1) if tokens else None,
                    "peakMemGB": None if DRY else round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "firstLoggedLoss": logs[0]["loss"] if logs else None, "lastLoggedLoss": logs[-1]["loss"] if logs else None}
    model = tr.model
    model.eval()
    model.config.use_cache = True
    evals["final"] = {"evalLoss": C.eval_loss(model, tok, HELD, DEV, batch_size=1, dtype=AMP)["loss"]}
    g = C.eval_generate(model, tok, GEN, DEV, dtype=AMP, max_new_tokens=GEN_TOKENS)
    evals["final"]["generation"] = {k: v for k, v in g.items() if k != "samples"}
    rec["samples"] = g["samples"]
    if run.get("rawPromptGen"):
        g2 = C.eval_generate(model, tok, GEN, DEV, dtype=AMP, template=False, max_new_tokens=GEN_TOKENS)
        evals["rawPrompt"] = {k: v for k, v in g2.items() if k != "samples"}
        rec["rawPromptSamples"] = g2["samples"][:6]
    if rcb.evals:
        evals["epochs"] = rcb.evals
    rec["evals"] = evals
    rec["wallClockSec"] = round(time.time() - t_run, 1)
    rec["date"] = time.strftime("%Y-%m-%d")
    return rec


for run in QUEUE:
    try:
        rec = one(run)
        rec["ok"] = True
    except Exception as e:
        rec = {"runId": run["runId"], "ok": False, "error": repr(e)[:800], "trace": traceback.format_exc()[-3000:], "config": run}
        print(rec["trace"], flush=True)
    with open(os.path.join(out_dir, run["runId"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("DONE", run["runId"], rec.get("ok"), rec.get("train"), (rec.get("evals") or {}).get("final", {}).get("evalLoss"), flush=True)
    gc.collect()
    if not DRY:
        torch.cuda.empty_cache()
