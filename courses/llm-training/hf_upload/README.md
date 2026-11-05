# Upload lab checkpoints to the Hugging Face Hub

The labs download our checkpoints from the model repo `aimlcompanion/llm-training-lab`, one folder per module. The files are not in git (`.gitignore` keeps `*.pt` out). Run these steps once per module script.

## lltTinyStories

1. Log in with a fine-grained token that has write access to the `aimlcompanion` org.

   ```
   pip install huggingface_hub
   huggingface-cli login
   ```

   Paste the token when asked. It is cached in your user profile, and the script reads it from there only.

2. Run the upload from this folder. The script first prints the sha256 and size of each file and stops if either differs from the file the lab was checked against.

   ```
   python upload_lltTinyStories.py --dry-run
   python upload_lltTinyStories.py
   ```

   | File | sha256 | Bytes |
   |---|---|---|
   | `step2000.pt` | `0d52171dc1ca2ce1f060b7a0ba4c5347c1c4c93c35de92c3da0ed854c84c753b` | 45,918,261 |
   | `tokenizer.json` | `b2f8a913ca18674c74262a43a6663fd5ea1f63211e72f18e3a81b47f7b9a03c1` | 265,249 |

3. Check that both URLs the script prints download in a browser, signed out:

   - https://huggingface.co/aimlcompanion/llm-training-lab/resolve/main/lltTinyStories/step2000.pt
   - https://huggingface.co/aimlcompanion/llm-training-lab/resolve/main/lltTinyStories/tokenizer.json

   The repo must be public, or the notebook's download fails for learners.
