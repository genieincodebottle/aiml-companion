"""Exact CPU measurements for the lltRewardModels page (no timings).

  python measure.py trlfilter  RewardTrainer at max_length 512 and 1,024 on the first 1,000 shuffled training pairs:
                             how many pairs it keeps, and what it prints -> run lltRewardModels-t1-trlfilter
  python measure.py data     length and tie statistics of the full UltraFeedback binarized preference splits,
                             counted with the reward model's chat template -> run lltRewardModels-t1-data
"""
import os, sys
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from runlog import write_run  # noqa: E402


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    tok.chat_template = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct").chat_template
    tok.pad_token = "<|endoftext|>"
    return tok


def data():
    from datasets import load_dataset
    tok = tokenizer()
    evals = {}
    for split in ("train_prefs", "test_prefs"):
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)

        def lengths(batch):
            out_c, out_r = [], []
            for p, c, r in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
                out_c.append(len(tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": c[-1]["content"]}])["input_ids"]))
                out_r.append(len(tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r[-1]["content"]}])["input_ids"]))
            return {"lc": out_c, "lr": out_r}
        L = ds.map(lengths, batched=True, batch_size=256, remove_columns=ds.column_names, num_proc=4)
        lc, lr = np.array(L["lc"]), np.array(L["lr"])
        sc, sr = np.array(ds["score_chosen"]), np.array(ds["score_rejected"])
        same = np.array([c[-1]["content"] == r[-1]["content"] for c, r in zip(ds["chosen"], ds["rejected"])])
        e = {"rows": len(ds), "scoreTieShare": float((sc == sr).mean()), "identicalReplies": int(same.sum()),
             "chosenLongerShare": float((lc > lr).mean()), "equalLengthShare": float((lc == lr).mean()),
             "meanChosenTokens": float(lc.mean()), "meanRejectedTokens": float(lr.mean()),
             "medianChosenTokens": float(np.median(lc)), "medianRejectedTokens": float(np.median(lr))}
        for cap in (512, 768, 1024, 2048):
            keep = (lc <= cap) & (lr <= cap)
            e["fitShare%d" % cap] = float(keep.mean())
            e["chosenLongerShareWithin%d" % cap] = float((lc[keep] > lr[keep]).mean())
        evals[split] = e
        print(split, e, flush=True)
    write_run("lltRewardModels-t1-data", "T1", "baseline",
              {"dataset": "HuggingFaceH4/ultrafeedback_binarized", "tokenizer": "SmolLM2-360M + SmolLM2-360M-Instruct chat template",
               "count": "tokens of the full chat (default system turn, user prompt, assistant reply)"}, 0, [], evals=evals)


def trlfilter():
    import contextlib, io, logging, torch
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForSequenceClassification
    from trl import RewardConfig, RewardTrainer
    torch.set_num_threads(4)
    tok = tokenizer()
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=0).select(range(1000))
    rows = [{"chosen": [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["chosen"][-1]["content"]}],
             "rejected": [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["rejected"][-1]["content"]}]}
            for ex in ds]
    model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM2-360M", num_labels=1, dtype=torch.float32)
    evals = {"pairsGiven": len(rows)}
    for L in (512, 1024):
        buf, logs = io.StringIO(), io.StringIO()
        h = logging.StreamHandler(logs); h.setLevel(logging.WARNING)
        logging.getLogger().addHandler(h); logging.getLogger("trl").addHandler(h); logging.getLogger("transformers").addHandler(h)
        with contextlib.redirect_stderr(buf):
            args = RewardConfig(output_dir="rm-filter-check", report_to="none", use_cpu=True, bf16=False, max_length=L)
            tr = RewardTrainer(model=model, args=args, processing_class=tok, train_dataset=Dataset.from_list(rows))
        for lg in (logging.getLogger(), logging.getLogger("trl"), logging.getLogger("transformers")):
            lg.removeHandler(h)
        err = buf.getvalue()
        evals["maxLength%d" % L] = {"kept": len(tr.train_dataset), "droppedShare": round(1 - len(tr.train_dataset) / len(rows), 4),
                                    "warningsLogged": logs.getvalue().strip()[:500],
                                    "progressBarSeen": ("Filtering" in err), "stderrTail": err.strip()[-300:]}
        print(L, evals["maxLength%d" % L], flush=True)
    import shutil
    shutil.rmtree("rm-filter-check", ignore_errors=True)
    write_run("lltRewardModels-t1-trlfilter", "T1", "baseline",
              {"trainer": "trl 1.17.1 RewardTrainer", "pairs": "first 1,000 of train_prefs shuffled with seed 0, conversational",
               "max_length": [512, 1024]}, 0, [], evals=evals)


if __name__ == "__main__":
    {"data": data, "trlfilter": trlfilter}[sys.argv[1]]()
