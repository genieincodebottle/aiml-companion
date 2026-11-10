"""Upload our reference adapters to the public Hub repo in one commit, under lltCapstoneReasoner/ only.

  python hf_upload.py <local folder with <run>/adapter-final/ and <run>/adapter-step50/> [--dry-run]

Uses the token cached by `hf auth login` (never written into this folder). Learners load an adapter with
  PeftModel.from_pretrained(base, "aimlcompanion/llm-training-lab", subfolder="lltCapstoneReasoner/<run>/adapter-final")
or with this lab's evaluate.py --adapter hub:lltCapstoneReasoner/<run>/adapter-final.
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO, PREFIX = "aimlcompanion/llm-training-lab", "lltCapstoneReasoner"
src = Path(sys.argv[1])
ops, manifest = [], {}
for f in sorted(src.rglob("*")):
    if not f.is_file() or not any(p.startswith("adapter-") for p in f.parts):
        continue
    if f.name not in ("adapter_model.safetensors", "adapter_config.json"):
        continue
    rel = PREFIX + "/" + f.relative_to(src).as_posix()
    manifest[rel] = dict(bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(f)))
assert all(o.path_in_repo.startswith(PREFIX + "/") for o in ops)
ops.append(CommitOperationAdd(path_in_repo=PREFIX + "/manifest.json", path_or_fileobj=json.dumps(manifest, indent=1).encode()))
for k, v in manifest.items():
    print(k, v["bytes"])
print(len(ops), "files,", round(sum(v["bytes"] for v in manifest.values()) / 1e6, 1), "MB")
if "--dry-run" not in sys.argv:
    info = HfApi().create_commit(repo_id=REPO, operations=ops, commit_message="lltCapstoneReasoner: reference GRPO adapters (Countdown)")
    print("committed", info.oid)
