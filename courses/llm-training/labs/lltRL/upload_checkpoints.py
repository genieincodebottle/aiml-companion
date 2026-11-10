"""Upload lltRL checkpoints to aimlcompanion/llm-training-lab under lltRL/<name>/, one file per commit, so a dropped
connection loses at most one file. Files already on the Hub with the same size are skipped, so rerunning resumes.

  python -I upload_checkpoints.py <local dir> <name> <README.md>
<local dir> holds a saved model (config.json, model.safetensors, generation_config.json) or a PEFT adapter
(adapter_config.json, adapter_model.safetensors). Uses the logged-in Hugging Face token (hf auth login); never prints it.
"""
import os, sys, time
from huggingface_hub import HfApi

local, name, readme = sys.argv[1:4]
REPO = "aimlcompanion/llm-training-lab"
api = HfApi()


def on_hub(prefix):
    try:
        return {f.path: getattr(f, "size", None) for f in api.list_repo_tree(REPO, path_in_repo=prefix, recursive=True)}
    except Exception:
        return {}


def push(path, dest, msg):
    have = on_hub(os.path.dirname(dest))
    if dest in have and have[dest] == os.path.getsize(path):
        print("on the Hub already", dest, flush=True)
        return
    for attempt in range(1, 9):
        try:
            t = time.time()
            api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=REPO, repo_type="model", commit_message=msg)
            print("uploaded", dest, os.path.getsize(path), "bytes in", round(time.time() - t), "s", flush=True)
            return
        except Exception as e:
            print("attempt", attempt, "failed for", dest, type(e).__name__, str(e)[:160], flush=True)
            time.sleep(min(300, 30 * attempt))
    raise SystemExit("gave up on " + dest)


for f in sorted(os.listdir(local)):
    if os.path.isfile(os.path.join(local, f)) and not f.startswith("README"):
        push(os.path.join(local, f), "lltRL/" + name + "/" + f, "lltRL: " + name + " (" + f + ")")
push(readme, "lltRL/" + name + "/README.md", "lltRL: " + name + " README")
print("DONE", sorted(p for p in on_hub("lltRL/" + name)), flush=True)
