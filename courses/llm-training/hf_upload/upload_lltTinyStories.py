"""Upload the lltTinyStories checkpoint files to the Hugging Face Hub.

Target: model repo aimlcompanion/llm-training-lab, folder lltTinyStories/
Files:  step2000.pt (our seed-0 baseline weights at step 2,000) and tokenizer.json

The token comes ONLY from your cached login (`huggingface-cli login`). This script never reads a
token from a file in the repo, never takes one as an argument, and never prints one.

Usage (from this folder):
    python upload_lltTinyStories.py            # upload
    python upload_lltTinyStories.py --dry-run  # print sha256 and size only
"""
import hashlib, os, sys
from huggingface_hub import HfApi

REPO_ID = "aimlcompanion/llm-training-lab"
FOLDER = "lltTinyStories"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "labs", "lltTinyStories", "checkpoint"))
FILES = ["step2000.pt", "tokenizer.json"]
EXPECTED = {
    "step2000.pt": ("0d52171dc1ca2ce1f060b7a0ba4c5347c1c4c93c35de92c3da0ed854c84c753b", 45918261),
    "tokenizer.json": ("b2f8a913ca18674c74262a43a6663fd5ea1f63211e72f18e3a81b47f7b9a03c1", 265249),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for name in FILES:
        path = os.path.join(SRC, name)
        digest, size = sha256(path), os.path.getsize(path)
        print(name, "sha256", digest, "size", size)
        if (digest, size) != EXPECTED[name]:
            sys.exit("STOP: " + name + " does not match the file the lab page and notebook were checked against.")
    if "--dry-run" in sys.argv:
        return
    api = HfApi()   # uses the token cached by `huggingface-cli login`
    user = api.whoami()
    print("logged in as", user.get("name"))
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    for name in FILES:
        api.upload_file(path_or_fileobj=os.path.join(SRC, name), path_in_repo=FOLDER + "/" + name,
                        repo_id=REPO_ID, repo_type="model",
                        commit_message="lltTinyStories: add " + name)
    for name in FILES:
        print("https://huggingface.co/" + REPO_ID + "/resolve/main/" + FOLDER + "/" + name)


if __name__ == "__main__":
    main()
