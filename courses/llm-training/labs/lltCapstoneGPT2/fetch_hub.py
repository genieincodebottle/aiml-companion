# fetch_hub.py - download our reference-run files from the Hub and check each one against manifest.json.
#
#   python fetch_hub.py step_004529_state.pt          # the quick path's starting checkpoint (about 1.5 GB)
#   python fetch_hub.py weights_2p5B.pt weights_1B.pt # our budget models, to compare with yours
#
# Files land in the current folder. No token is needed: the repo is public.
import hashlib, json, os, sys
from huggingface_hub import hf_hub_download

REPO, FOLDER = "aimlcompanion/llm-training-lab", "lltCapstoneGPT2"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


manifest = json.load(open(hf_hub_download(REPO, f"{FOLDER}/manifest.json")))
for name in sys.argv[1:] or ["step_004529_state.pt"]:
    if name not in manifest["files"]:
        raise SystemExit(f"{name} is not in the manifest; files: {', '.join(sorted(manifest['files']))}")
    want = manifest["files"][name]
    # local_dir writes the file once into ./lltCapstoneGPT2/ (no second copy in the cache), then we move it here
    path = hf_hub_download(REPO, f"{FOLDER}/{name}", local_dir=".")
    got = sha256(path)
    if got != want["sha256"]:
        raise SystemExit(f"{name}: sha256 {got[:12]} does not match the manifest {want['sha256'][:12]}; delete it and retry")
    os.replace(path, os.path.basename(name))
    try:
        os.removedirs(os.path.dirname(path))      # drop the now-empty ./lltCapstoneGPT2/... folders
    except OSError:
        pass
    print(f"{name}: {want['bytes'] / 1e9:.3f} GB, sha256 matches the manifest")
