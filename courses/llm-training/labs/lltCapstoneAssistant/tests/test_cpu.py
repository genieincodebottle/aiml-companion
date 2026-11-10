"""CPU tests for the capstone pipeline. Run from the lab folder: python -m pytest -q tests/test_cpu.py  (or python tests/test_cpu.py without pytest)

They use the real Qwen3-1.7B-Base tokenizer (a small download) and a tiny random model with the same vocabulary,
so they check the parts that fail silently: the label mask, the end-of-text label, packing, the pair builder,
the verdict parser and the statistics.
"""
import math
import os
import sys

import numpy as np
try:
    import pytest
except ImportError:                      # the runner at the bottom works without pytest
    class pytest:
        @staticmethod
        def fixture(**kw):
            return lambda f: f
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capstone.common import BASE, CHAT_TEMPLATE, EOS, TEMPLATE, ngrams, norm  # noqa: E402
from capstone.label import build_pairs  # noqa: E402
from capstone.judge import VERDICT  # noqa: E402
from capstone.scores import refused  # noqa: E402
from capstone.stats import boot, lc_winrate, paired_diff, wilson  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from capstone.train_common import tokenizer
    return tokenizer(BASE)


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).float()


def test_template_and_eos(tok):
    assert tok.eos_token == EOS and tok.eos_token_id == 151643 and tok.pad_token_id == tok.eos_token_id
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
    assert tok.apply_chat_template(msgs, tokenize=False) == TEMPLATE.format("Hi") + "Hello." + EOS
    assert CHAT_TEMPLATE.count("endoftext") == 1


def sft_rows():
    return [{"prompt": TEMPLATE.format("Name a colour."), "completion": "Blue."},
            {"prompt": TEMPLATE.format("Add 2 and 3."), "completion": "2 + 3 = 5."},
            {"prompt": TEMPLATE.format("Say hi in French."), "completion": "Bonjour."}]


def make_sft(tok, tiny, tmp_path, packing):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    args = SFTConfig(output_dir=str(tmp_path), max_length=128, packing=packing, packing_strategy="bfd", completion_only_loss=True,
                     eos_token=tok.eos_token, pad_token=tok.pad_token, per_device_train_batch_size=3, report_to="none",
                     use_cpu=True, max_steps=1)
    return SFTTrainer(model=tiny, args=args, train_dataset=Dataset.from_list(sft_rows()), processing_class=tok)


def test_sft_mask_ends_with_eos(tok, tiny, tmp_path):
    tr = make_sft(tok, tiny, tmp_path / "a", packing=False)
    row = tr.train_dataset[0]
    ids, lab = row["input_ids"], row["labels"]
    assert ids[-1] == tok.eos_token_id, "the reply must end with <|endoftext|> or the model never learns to stop"
    assert lab[-1] == tok.eos_token_id, "the final EOS must carry a label"
    n_prompt = len(tok(TEMPLATE.format("Name a colour.")).input_ids)
    assert all(x == -100 for x in lab[:n_prompt]) and lab[n_prompt:] == ids[n_prompt:], "prompt masked, reply trained"


def test_packing_matches_unpacked_loss(tok, tiny, tmp_path):
    """Packed rows with a per-row position reset must give the same loss as the rows run one by one."""
    tr = make_sft(tok, tiny, tmp_path / "b", packing=True)
    batch = tr.data_collator([tr.train_dataset[i] for i in range(len(tr.train_dataset))])
    assert batch["input_ids"].shape[0] == 1 and "position_ids" in batch
    tiny.eval()
    with torch.no_grad():
        packed = tiny(input_ids=batch["input_ids"], position_ids=batch["position_ids"]).logits
    labels = batch["labels"][0]
    lp = torch.log_softmax(packed[0, :-1].float(), -1)
    tgt = labels[1:]
    keep = tgt != -100
    packed_nll = -lp[keep].gather(1, tgt[keep][:, None]).sum()
    single = 0.0
    for r in sft_rows():
        p = tok(r["prompt"]).input_ids
        full = tok(r["prompt"] + r["completion"]).input_ids + [tok.eos_token_id]
        with torch.no_grad():
            lg = tiny(input_ids=torch.tensor([full])).logits[0, :-1].float()
        lps = torch.log_softmax(lg, -1)
        t = torch.tensor(full[1:])
        single += -lps[torch.arange(len(p) - 1, len(full) - 1), t[len(p) - 1:]].sum()
    assert torch.allclose(packed_nll, single, atol=1e-3), (packed_nll.item(), float(single))


def test_dpo_rows_end_with_eos(tok, tiny, tmp_path):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    from peft import LoraConfig
    rows = [{"prompt": TEMPLATE.format("Name a colour."), "chosen": "Blue.", "rejected": "I like cars."}] * 2
    args = DPOConfig(output_dir=str(tmp_path / "d"), max_length=128, report_to="none", use_cpu=True, max_steps=1,
                     per_device_train_batch_size=2, pad_token=tok.pad_token)
    tr = DPOTrainer(model=tiny, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok,
                    peft_config=LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    r = tr.train_dataset[0]
    keys = [k for k in r if "chosen" in k and "ids" in k]
    ids = r[keys[0]]
    assert ids[-1] == tok.eos_token_id, (keys, ids[-5:])
    tr.train()


def test_pairs():
    rows = [{"id": "p", "prompt": "q", "replies": [{"text": "a", "tokens": 3, "finished": True, "rm": 2.0},
                                                    {"text": "b", "tokens": 9, "finished": False, "rm": 5.0},
                                                    {"text": "c", "tokens": 4, "finished": True, "rm": -1.0}]},
            {"id": "t", "prompt": "q2", "replies": [{"text": "x", "tokens": 3, "finished": True, "rm": 1.0},
                                                     {"text": "y", "tokens": 3, "finished": True, "rm": 1.5}]}]
    pairs, skipped = build_pairs(rows, "rm", 1.0)
    assert len(pairs) == 1 and pairs[0]["chosen"] == "a" and pairs[0]["rejected"] == "c"   # the capped reply never wins
    assert skipped["gap below min"] == 1


def test_verdict_parser_takes_last():
    text = "If A were wrong it would be [[B]], but A is right.\nFinal: [[A]]"
    assert VERDICT.findall(text)[-1] == "A"


def test_refusal_rule():
    assert refused("I can’t help with that.") and refused("Sorry, no.") and not refused("Sure. Here is how.")


def test_decontamination_ngrams():
    a = "Write a 300+ word summary of the wikipedia page about Raymond III, Count of Tripoli, in plain words please now"
    assert ngrams(a) & ngrams("x " + a + " y")
    assert norm("Hello,  World!") == "hello world"


def test_stats():
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    lo, hi = boot([1, 0] * 50)
    assert lo < 0.5 < hi
    d = paired_diff([1, 1, 0, 1], [0, 1, 0, 0])
    assert d["diff"] == 0.5
    rng = np.random.default_rng(0)
    dl = rng.normal(0, 200, 4000)
    p = 1 / (1 + np.exp(-(-0.4 + 1.5 * np.tanh(dl / dl.std()))))
    y = (rng.random(4000) < p).astype(float)
    r = lc_winrate(y, dl, n_boot=50)
    assert abs(r["lc"] - 1 / (1 + math.exp(0.4))) < 0.03 and r["lengthCoef"] > 1.0


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    t, m = tok(), None
    m = tiny(t)
    tmp = Path(tempfile.mkdtemp())
    for f, args in [(test_template_and_eos, (t,)), (test_sft_mask_ends_with_eos, (t, m, tmp)),
                    (test_packing_matches_unpacked_loss, (t, m, tmp)), (test_dpo_rows_end_with_eos, (t, m, tmp)),
                    (test_pairs, ()), (test_verdict_parser_takes_last, ()), (test_refusal_rule, ()),
                    (test_decontamination_ngrams, ()), (test_stats, ())]:
        f(*args)
        print("PASS", f.__name__, flush=True)
