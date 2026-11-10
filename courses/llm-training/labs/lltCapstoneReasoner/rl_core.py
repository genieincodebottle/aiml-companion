"""The GRPO building blocks of the lltRL lab, loaded from its step files so there is one version of them.

We do not copy the code. We read these functions out of ../lltRL/steps/*.py by name and run them here:
  kl_k3                              (cell02, the k3 KL estimator)
  autocast                           (cell04, fp16 autocast on the GPU)
  group_advantages, rloo_advantages  (cell07, GRPO and RLOO advantages)
  grpo_loss, completion_logps        (cell08, the clipped GRPO loss and the per-token log-probs)
  load_policy, sample, pass_at_k     (cell09, frozen fp16 base + fp32 LoRA r16, sampling with stop strings,
                                      the unbiased pass@k estimator)
The capstone adds the Countdown task (countdown.py), a resumable loop (train.py) and the evaluation (evaluate.py).
Each run file records the sha256 of every function it used, so a later change in lltRL is visible.

Usage:  import rl_core; ns = rl_core.load(model_id, max_new, stops, tokenizer)
"""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = Path(__import__("os").environ.get("LLTRL_STEPS", HERE.parent / "lltRL" / "steps"))
WANT = {
    "cell02_step2.py": ["kl_k3"],
    "cell04_step4.py": ["autocast"],
    "cell07_step7.py": ["group_advantages", "rloo_advantages"],
    "cell08_step8.py": ["grpo_loss", "completion_logps"],
    "cell09_step9.py": ["load_policy", "sample", "pass_at_k"],
}


def extract(path, names):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError("lltRL step file " + str(path) + " no longer defines " + ", ".join(sorted(missing)))
    return found


def sources():
    out = {}
    for fname, names in WANT.items():
        out.update(extract(STEPS / fname, names))
    return out


def hashes(src=None):
    src = src or sources()
    return {k: hashlib.sha256(v.encode("utf-8")).hexdigest()[:16] for k, v in sorted(src.items())}


def load(model_id, max_new, stops, tokenizer, dev=None):
    """Execute the lltRL functions in a fresh namespace with the globals they expect, and return it."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, GenerationConfig
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns = dict(torch=torch, math=__import__("math"), AutoModelForCausalLM=AutoModelForCausalLM, GenerationConfig=GenerationConfig,
              LoraConfig=LoraConfig, PeftModel=PeftModel, get_peft_model=get_peft_model,
              DEV=dev, AMP=dev.type == "cuda", GMODEL=model_id, MAX_NEW=max_new, STOPS=list(stops), qtok=tokenizer, NL=chr(10))
    src = sources()
    for name, code in src.items():
        exec(compile(code, "lltRL:" + name, "exec"), ns)
    ns["_hashes"] = hashes(src)
    ns["_sources"] = src
    return ns
