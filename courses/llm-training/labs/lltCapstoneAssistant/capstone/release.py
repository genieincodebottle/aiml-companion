"""Stage 6b. Release: reshard the merged model, then upload it with its card, one file per commit.

  python -m capstone.release prepare --work <dir>          writes <work>/release/dpo-merged (500 MB shards, card, tokenizer)
                                                             and <work>/release/sft-adapter, <work>/release/data
  python -m capstone.release upload --src <release dir> --repo <you>/<repo> [--prefix <folder>/]
One file per commit, skipping files already on the Hub with the same size, so a dropped connection costs one
file and a rerun resumes. Uses the logged-in Hugging Face token (hf auth login); it never prints it.
"""
import os
import shutil
import sys
import time

from .common import cli, path


def prepare(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = path(a, "release", "dpo-merged", "x")[:-2]
    if not os.path.exists(os.path.join(out, "model.safetensors.index.json")):
        src = path(a, "dpo", "merged", "x")[:-2]
        m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
        m.save_pretrained(out, max_shard_size="500MB")
        AutoTokenizer.from_pretrained(src).save_pretrained(out)
        for f in ("README.md", "generation_config.json"):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), out)
    ad = path(a, "release", "sft-adapter", "x")[:-2]
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(path(a, "sft", "adapter", f), ad)
    dd = path(a, "release", "data", "x")[:-2]
    shutil.copy(path(a, "pairs", "pairs.jsonl.gz"), dd)
    print("release folder ready:", path(a, "release", "x")[:-2])


def upload(src, repo, prefix):
    from huggingface_hub import HfApi
    api = HfApi()

    def on_hub(folder):
        try:
            return {f.path: getattr(f, "size", None) for f in api.list_repo_tree(repo, path_in_repo=folder, recursive=True)}
        except Exception:
            return {}
    for root, _, files in os.walk(src):
        for f in sorted(files):
            local = os.path.join(root, f)
            rel = os.path.relpath(local, src).replace(os.sep, "/")
            dest = prefix + rel
            have = on_hub(os.path.dirname(dest))
            if have.get(dest) == os.path.getsize(local):
                print("on the Hub already", dest, flush=True)
                continue
            for attempt in range(1, 9):
                try:
                    t = time.time()
                    api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=repo, repo_type="model",
                                    commit_message="lltCapstoneAssistant: " + rel)
                    print("uploaded", dest, os.path.getsize(local), "bytes in", round(time.time() - t), "s", flush=True)
                    break
                except Exception as e:
                    print("attempt", attempt, "failed for", dest, type(e).__name__, str(e)[:160], flush=True)
                    time.sleep(min(300, 30 * attempt))
            else:
                raise SystemExit("gave up on " + dest)


def main():
    mode = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in ("prepare", "upload") else "prepare"
    a = cli("Stage 6b: release", [(("--src",), dict(default=None)), (("--repo",), dict(default="aimlcompanion/llm-training-lab")),
                                  (("--prefix",), dict(default="lltCapstoneAssistant/"))])
    if mode == "prepare":
        prepare(a)
    else:
        upload(a.src or path(a, "release", "x")[:-2], a.repo, a.prefix)


if __name__ == "__main__":
    main()
