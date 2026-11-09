"""Upload the lltDPO checkpoints to aimlcompanion/llm-training-lab, one file per commit, so a dropped connection
loses at most one file. Files already on the Hub with the same size are skipped, so rerunning resumes.

  python -I upload_checkpoints.py <kernel out dir with sft-model/ and dpo-b0.1-s0/> <README_sft.md> <README_dpo.md>
lltDPO/dpo-b0.1-s0/  LoRA adapter of the main DPO run (adapter_config.json, adapter_model.safetensors) + README
lltDPO/sft-s0/       merged SFT model, resharded locally into files of at most 200 MB (same tensors, same values),
                     + config, generation config, tokenizer files and README
Uses the logged-in Hugging Face token (hf auth login).
"""
import os, shutil, sys, time
from huggingface_hub import HfApi

out, readme_sft, readme_dpo = sys.argv[1:4]
REPO = "aimlcompanion/llm-training-lab"
api = HfApi()


def on_hub(prefix):
    try:
        return {f.path: getattr(f, "size", None) for f in api.list_repo_tree(REPO, path_in_repo=prefix, recursive=True)}
    except Exception:
        return {}


def push(local, dest, msg):
    have = on_hub(os.path.dirname(dest))
    if dest in have and have[dest] == os.path.getsize(local):
        print("on the Hub already", dest, flush=True)
        return
    for attempt in range(1, 9):
        try:
            t = time.time()
            api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=REPO, repo_type="model", commit_message=msg)
            print("uploaded", dest, os.path.getsize(local), "bytes in", round(time.time() - t), "s", flush=True)
            return
        except Exception as e:
            print("attempt", attempt, "failed for", dest, type(e).__name__, str(e)[:160], flush=True)
            time.sleep(min(300, 30 * attempt))
    raise SystemExit("gave up on " + dest)


# 1. the small adapter first
ad = os.path.join(out, "dpo-b0.1-s0")
for f in ["adapter_config.json", "adapter_model.safetensors"]:
    push(os.path.join(ad, f), "lltDPO/dpo-b0.1-s0/" + f, "lltDPO: DPO LoRA adapter, beta 0.1, seed 0 (" + f + ")")
push(readme_dpo, "lltDPO/dpo-b0.1-s0/README.md", "lltDPO: adapter README")

# 2. the SFT model, resharded into files of at most 200 MB
src = os.path.join(out, "sft-model")
shard_dir = os.path.join(out, "sft-model-sharded")
if not os.path.exists(os.path.join(shard_dir, "model.safetensors.index.json")):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float16)
    m.save_pretrained(shard_dir, max_shard_size="200MB")
    AutoTokenizer.from_pretrained(src).save_pretrained(shard_dir)
    del m
files = sorted(f for f in os.listdir(shard_dir) if os.path.isfile(os.path.join(shard_dir, f)))
for f in files:
    push(os.path.join(shard_dir, f), "lltDPO/sft-s0/" + f, "lltDPO: SFT model, merged LoRA (" + f + ")")
push(readme_sft, "lltDPO/sft-s0/README.md", "lltDPO: SFT model README")
print("DONE", sorted(on_hub("lltDPO")), flush=True)
