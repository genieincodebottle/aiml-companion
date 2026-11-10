import tempfile
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

def grpo_loss(new_lp, old_lp, adv, mask, eps=0.2, eps_high=None, beta=0.0, ref_lp=None, loss_type="dapo", max_len=None, denom=None):
    # new_lp, old_lp, ref_lp, mask: [replies, completion tokens]; adv: one number per reply.
    # denom: the normalizer of the whole minibatch when you accumulate micro-batches (None = this batch alone)
    hi = eps if eps_high is None else eps_high                       # DAPO's clip-higher raises only the upper bound
    ratio = torch.exp(new_lp - old_lp)
    per_tok = -torch.min(ratio * adv[:, None], torch.clamp(ratio, 1 - eps, 1 + hi) * adv[:, None])
    if beta:
        per_tok = per_tok + beta * kl_k3(new_lp, ref_lp)            # k3 KL to the reference, added per token
    if loss_type == "grpo":                                          # mean over each reply's tokens, then over replies
        return ((per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)).sum() / (denom or mask.shape[0])
    if loss_type == "dr_grpo":                                       # divide by a constant, replies x max length
        return (per_tok * mask).sum() / (denom or mask.shape[0] * max_len)
    return (per_tok * mask).sum() / (denom or mask.sum().clamp(min=1))   # "dapo": every token in the batch counts once

def completion_logps(model, prompt_ids, prompt_mask, completion_ids, completion_mask):
    ids = torch.cat([prompt_ids, completion_ids], 1)
    att = torch.cat([prompt_mask, completion_mask], 1)
    # logits_to_keep: compute the vocabulary logits only for the last C + 1 positions, the ones that predict the reply
    logits = model(input_ids=ids, attention_mask=att, logits_to_keep=completion_ids.shape[1] + 1).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, completion_ids[..., None]).squeeze(-1)

# One fixed batch: 2 prompts x 4 replies, left-padded prompts, right-padded replies, as GRPOTrainer builds them
torch.manual_seed(0)
SMALL = "HuggingFaceTB/SmolLM2-135M"            # the loss does not depend on the model, so a small one keeps this on a CPU
gtok = AutoTokenizer.from_pretrained(SMALL, padding_side="left")
gtok.pad_token = gtok.pad_token or gtok.eos_token
cpu_policy = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32).eval()
enc = gtok(["Question: What is 6 x 7?" + NL + "Answer:"] * 4 + ["Question: Add 15 and 27." + NL + "Answer:"] * 4, return_tensors="pt", padding=True)
completion_ids = torch.randint(100, 5000, (8, 12))
completion_mask = (torch.arange(12)[None] < torch.tensor([12, 9, 12, 5, 7, 12, 3, 10])[:, None]).long()
rewards = torch.tensor([1., 0., 0., 1., 0., 0., 0., 1.])
adv = group_advantages(rewards, 4)
with torch.no_grad():
    lp = completion_logps(cpu_policy, enc.input_ids, enc.attention_mask, completion_ids, completion_mask)
old_lp = lp + 0.3 * torch.randn_like(lp)          # pretend the sampling policy differed a little, so the clip matters
ref_lp = lp + 0.2 * torch.randn_like(lp)

ds = Dataset.from_dict({"prompt": ["x"] * 8})
for loss_type, beta, eps_high in [("dapo", 0.0, None), ("grpo", 0.04, None), ("dr_grpo", 0.0, 0.28)]:
    args = GRPOConfig(output_dir=tempfile.mkdtemp(), use_cpu=True, per_device_train_batch_size=8, num_generations=4,
                      max_completion_length=12, loss_type=loss_type, beta=beta, epsilon_high=eps_high, bf16=False, report_to="none")
    trainer = GRPOTrainer(model=cpu_policy, reward_funcs=lambda completions, **kw: [0.0] * len(completions), args=args,
                          train_dataset=ds, processing_class=gtok)
    trainer.model.eval()                            # eval mode: TRL skips the gradient-accumulation scaling
    inputs = dict(prompt_ids=enc.input_ids, prompt_mask=enc.attention_mask, completion_ids=completion_ids, completion_mask=completion_mask,
                  advantages=adv, old_per_token_logps=old_lp, ref_per_token_logps=ref_lp, num_items_in_batch=completion_mask.sum())
    with torch.no_grad():
        theirs = trainer._compute_loss(trainer.model, inputs).item()
        new_lp = completion_logps(cpu_policy, enc.input_ids, enc.attention_mask, completion_ids, completion_mask)
        ours = grpo_loss(new_lp, old_lp, adv, completion_mask, 0.2, eps_high, beta, ref_lp, loss_type, 12).item()
    print(loss_type, "| beta", beta, "| epsilon_high", eps_high, "| ours", round(ours, 6), "| GRPOTrainer", round(theirs, 6), "| difference", abs(ours - theirs))
print("TRL GRPOConfig defaults: loss_type", GRPOConfig.loss_type, "| beta", GRPOConfig.beta, "| num_generations", GRPOConfig.num_generations,
      "| epsilon", GRPOConfig.epsilon, "| scale_rewards", GRPOConfig.scale_rewards, "| learning_rate", GRPOConfig.learning_rate)
