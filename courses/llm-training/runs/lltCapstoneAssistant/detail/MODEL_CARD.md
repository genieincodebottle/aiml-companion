---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B-Base
library_name: transformers
pipeline_tag: text-generation
tags: [capstone, sft, dpo, lora, llm-training-lab]
---

# lltCapstoneAssistant: an instruction-following assistant from Qwen/Qwen3-1.7B-Base

The reference model of the LLM Training Lab capstone. It starts from the **base** model Qwen/Qwen3-1.7B-Base and is trained in two
stages: supervised fine-tuning (SFT) on 20,000 single-turn conversations, then direct preference optimization
(DPO) on 4,593 preference pairs sampled from the SFT model itself and labelled by the reward model Skywork/Skywork-Reward-V2-Qwen3-8B.
Both stages train a LoRA adapter (rank 64) that is merged into the weights here.

It is a teaching model, built to show the method and the evaluation. It is weaker than Qwen/Qwen3-1.7B, Qwen's own post-trained model
from the same base (table below). Do not use it where mistakes cost anything.

## Prompt format

Single turn only. The base model's chat tokens are untrained, so the model uses a plain-text template and ends every reply
with `<|endoftext|>`. `tokenizer.apply_chat_template` builds it for you.

```
### Instruction:
{}

### Response:
<your reply is generated here><|endoftext|>
```

## Training data

- SFT: 20,000 conversations (11,350,253 tokens) sampled from allenai/Dolci-Instruct-SFT (ODC-BY; each
  source keeps its own licence, listed on that dataset's card). Mix by group: precise_if 3,992, chat 7,416, oasst 798, code 2,393, flan 1,005, science 600, safety 1,392, math 2,404.
  Single-turn English only, no tool calls, replies that name a model or lab removed, and every prompt that shares a
  13-word span with an IFEval, Arena-Hard v2.0 or XSTest prompt removed (1213 dropped).
  The WildChat replies in that set were written by GPT-4.1, and its Precise IF subset trains the same kinds of checkable
  constraints IFEval tests, so part of the IFEval gain reflects matched constraint types.
- DPO: 5,000 prompts from HuggingFaceH4/ultrafeedback_binarized (MIT; prompts only), 4 replies
  each from the SFT model, scored by Skywork/Skywork-Reward-V2-Qwen3-8B. Chosen = the best reply that ended on its own, rejected = the worst, kept when the
  score gap was at least 1.0. Chosen replies were longer in 61.5% of pairs.

## Evaluation

IFEval prompt-level strict accuracy (541 prompts, greedy, 95% bootstrap interval) and XSTest refusal rates on
200 unsafe and 250 safe prompts, first 64 tokens, as string match / LLM refusal judge (both checked against
XSTest's 2,250 human labels; the string rule misses refusals that lecture instead of apologising):

| Model | IFEval prompt strict | Refuses unsafe | Refuses safe (over-refusal) |
|---|---|---|---|
| base | 38.6% (34.6 to 42.7) | n/a | n/a |
| sft | 54.5% (50.3 to 58.6) | 89.5% / 96.5% | 17.6% / 24.5% |
| dpo | 53.4% (49.2 to 57.9) | 44.5% / 88.0% | 0.8% / 11.2% |
| official | 68.8% (64.9 to 72.6) | 57.0% / 88.0% | 4.8% / 21.2% |

Pairwise win rate on 500 Arena-Hard v2.0 prompts, judged by openai/gpt-oss-20b (open weights, run locally) in both
orders. Length-controlled = the win rate predicted at equal reply length (a one-term version of AlpacaEval's LC).

| Comparison | Win rate (95% CI) | Length-controlled | Mean reply length |
|---|---|---|---|
| dpo vs official | 52.6% (49.4 to 55.9) | 51.6% (47.9 to 55.5) | 723.5 / 910.7 tokens |
| sft vs official | 38.6% (35.5 to 41.6) | 38.5% (35.2 to 42.2) | 728.3 / 910.7 tokens |
| dpo vs sft | 63.5% (60.6 to 66.5) | 63.5% (60.5 to 66.4) | 723.5 / 728.3 tokens |

The judge was checked first on 300 HelpSteer3 pairs with clear human preferences: it agreed with the human label in
70.8% (95% CI 66.8 to 74.7) and gave the same verdict in both orders on 62.0% of pairs.

## Limitations

Written from our own evaluation of this model (run records in the repository linked below).

- **Weaker than Qwen3-1.7B at following instructions.** IFEval prompt-level strict accuracy is about 15 points below Qwen3-1.7B with thinking off, and DPO did not change IFEval compared with our SFT model (the paired interval includes zero).
- **Constraint types it often misses.** Against our SFT model, DPO passed fewer case constraints (all lowercase or all capitals) and fewer punctuation constraints (no commas). Examples: IFEval keys 1122 and 1153 ask for lowercase and the reply opens with capitals.
- **Loops.** 24 of 541 greedy IFEval replies ran to the 1,536-token cap, usually repeating one sentence (IFEval keys 1108 and 1389).
- **Refuses less than our SFT model.** On XSTest's unsafe prompts the LLM refusal judge counts 88.0% refusals against 96.5% for SFT, a paired drop of 8.5 points (4.5 to 13.0). Many refusals became warnings or lectures instead of a short decline. Over-refusal of safe prompts also fell. Do not rely on this model to decline harmful requests.
- **Reasoning and arithmetic.** It answers reasoning puzzles confidently and wrongly (Arena-Hard uid 4a275a24, average siblings).
- **Identity.** It has no name and may invent one when asked (Arena-Hard uid 047bd685).
- **Single turn and English.** Trained on single-turn English conversations only; multi-turn chat and other languages are untested beyond what the base model already does.
- **The judge is imperfect.** Win rates come from gpt-oss-20b, which agreed with HelpSteer3 human labels on about 71% of pairs and gave the same verdict in both orders on 62%.
- **Training data provenance.** Part of the SFT data are GPT-4.1 replies (Dolci WildChat), and the preference labels come from Skywork-Reward-V2-Qwen3-8B, so the model inherits the style and biases of both.


## How it was made

Code, runbook and every run file: https://github.com/genieincodebottle/aiml-companion/tree/main/courses/llm-training/labs/lltCapstoneAssistant
