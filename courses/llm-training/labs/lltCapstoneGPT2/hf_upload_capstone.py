# hf_upload_capstone.py - upload the reference run's learner files to the public Hub repo, one commit.
#
#   hf auth login                                   # paste a fine-grained WRITE token for aimlcompanion/llm-training-lab;
#                                                   # it is cached by huggingface_hub, never written into this folder
#   python hf_upload_capstone.py --dry-run          # list what would go up, with sizes and sha256
#   python hf_upload_capstone.py                    # upload
#
# Repo layout: aimlcompanion/llm-training-lab/lltCapstoneGPT2/<file>. Learner notebooks download from
# https://huggingface.co/aimlcompanion/llm-training-lab/resolve/main/lltCapstoneGPT2/<file> and check manifest.json.
import argparse, glob, hashlib, json, os, time
from huggingface_hub import CommitOperationAdd, HfApi

p = argparse.ArgumentParser()
p.add_argument("--workdir", default=os.environ.get("WORKDIR", "/workspace"))
p.add_argument("--repo", default="aimlcompanion/llm-training-lab")
p.add_argument("--folder", default="lltCapstoneGPT2")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
W = a.workdir


def one(pattern):
    hits = sorted(glob.glob(os.path.join(W, pattern)))
    if not hits:
        raise SystemExit("missing: " + os.path.join(W, pattern))
    return hits[-1]


# repo name -> local file. Only tensors and plain Python types: every file loads with torch.load(weights_only=True).
FILES = {
    "step_004529_state.pt": one("run/ckpt/step_004529/state.pt"),       # free path: full training state, end of stable phase
    "weights_10B_final.pt": one("run/weights/weights_step_019073.pt"),   # the main run's final model
    "weights_0p5B.pt": one("branches/0p5B-d5/weights/weights_step_*.pt"),
    "weights_1B.pt": one("branches/1B-d5/weights/weights_step_*.pt"),
    "weights_2p5B.pt": one("branches/2p5B-d5/weights/weights_step_*.pt"),  # the free path's target model
    "weights_2p5B_d10.pt": one("branches/2p5B-d10/weights/weights_step_*.pt"),
    "weights_5B.pt": one("branches/5B-d5/weights/weights_step_*.pt"),
    "runs/lltCapstoneGPT2-ref-124m-10B.json": one("run/lltCapstoneGPT2-ref-124m-10B.json"),
}
# Keep the subfolder (2p5B-d5/, curve/, ...): two branches both end at step 4,769, and the HellaSwag curve also scores
# step 19,073, so base names alone would collide and one file would silently replace another.
E = os.path.join(W, "run", "evals")
for f in sorted(glob.glob(os.path.join(E, "**", "*.json"), recursive=True)):
    FILES["runs/evals/" + os.path.relpath(f, E).replace(os.sep, "/")] = f
for f in sorted(glob.glob(os.path.join(W, "branches", "*", "lltCapstoneGPT2-ref-124m-*.json"))):
    FILES["runs/" + os.path.basename(f)] = f


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


manifest = {"repo": a.repo, "folder": a.folder, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loadWith": "torch.load(path, map_location='cpu', weights_only=True)", "files": {}}
for name, path in FILES.items():
    manifest["files"][name] = {"bytes": os.path.getsize(path), "sha256": sha256(path), "source": os.path.relpath(path, W)}
    print(f"{name:<48} {os.path.getsize(path) / 1e9:7.3f} GB  {manifest['files'][name]['sha256'][:12]}")
man_path = os.path.join(W, "run", "hub_manifest.json")
with open(man_path, "w") as fh:
    json.dump(manifest, fh, indent=1)
if a.dry_run:
    raise SystemExit("dry run: nothing uploaded; manifest written to " + man_path)

ops = [CommitOperationAdd(path_in_repo=f"{a.folder}/{name}", path_or_fileobj=path) for name, path in FILES.items()]
ops.append(CommitOperationAdd(path_in_repo=f"{a.folder}/manifest.json", path_or_fileobj=man_path))
api = HfApi()                                   # uses the token cached by `hf auth login`
info = api.create_commit(repo_id=a.repo, repo_type="model", operations=ops,
                         commit_message="lltCapstoneGPT2: reference run checkpoints and Run Library files")
print("uploaded:", info.commit_url)
