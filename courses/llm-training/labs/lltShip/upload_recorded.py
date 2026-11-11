"""Push recorded.json (built by make_recorded.py) to aimlcompanion/llm-training-lab under lltShip/. Uses the logged-in
Hugging Face token (hf auth login); never prints it. Only this one file is touched.

    python upload_recorded.py
"""
from pathlib import Path
from huggingface_hub import HfApi

here = Path(__file__).resolve().parent
print(HfApi().upload_file(path_or_fileobj=str(here / "recorded.json"), path_in_repo="lltShip/recorded.json",
                          repo_id="aimlcompanion/llm-training-lab", repo_type="model",
                          commit_message="lltShip: recorded A100 results for the FROM_OUR_CHECKPOINT exit (recorded.json)"))
