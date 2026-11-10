"""Start from our checkpoint: download our SFT adapter (or DPO pairs) from the Hub so you can skip a long stage.

  python -m capstone.fetch --what sft     our SFT LoRA adapter, merged into the base -> <work>/sft/merged
  python -m capstone.fetch --what pairs   our 4,000+ on-policy pairs -> <work>/pairs/pairs.jsonl

Our adapter was trained at T3 (rank 64, 20,000 conversations). Say in your report which stages you ran yourself.
"""
import gzip
import os
import shutil

from .common import BASE, cli, done, path

REPO = "aimlcompanion/llm-training-lab"
SUB = "lltCapstoneAssistant"


def main():
    a = cli("fetch our checkpoints", [(("--what",), dict(choices=["sft", "pairs"], required=True))])
    from huggingface_hub import hf_hub_download, snapshot_download
    if a.what == "sft":
        if done(path(a, "sft", "merged", "config.json")):
            print("sft/merged exists")
            return
        d = snapshot_download(REPO, allow_patterns=[SUB + "/sft-adapter/*"])
        from .train_common import merge_and_save
        cfg = dict(a.cfg)
        cfg["dtype"] = "float16" if a.tier == "T2" else "bfloat16"
        merge_and_save(BASE, os.path.join(d, SUB, "sft-adapter"), path(a, "sft", "merged", "x")[:-2], cfg)
        print("merged our SFT adapter into", path(a, "sft", "merged", "x")[:-2])
    else:
        f = hf_hub_download(REPO, SUB + "/data/pairs.jsonl.gz")
        with gzip.open(f, "rb") as src, open(path(a, "pairs", "pairs.jsonl"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        print("wrote", path(a, "pairs", "pairs.jsonl"))


if __name__ == "__main__":
    main()
