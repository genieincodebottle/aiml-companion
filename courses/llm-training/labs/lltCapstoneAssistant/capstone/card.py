"""Stage 6a. Write the model card (README.md) next to the merged DPO model, every number read from the stage records.

A model card says what the model is, what it was trained on, how it was evaluated, and where it fails
(Mitchell et al. 2019). Write the limitations from your own evaluation, not from a template.
"""
from .common import BASE, JUDGE, OFFICIAL, RM, TEMPLATE, cli, path, read_json, write_json


def pct(x):
    return "%.1f%%" % (100 * x)


def ci(c):
    return "%.1f to %.1f" % (100 * c[0], 100 * c[1])


def main():
    a = cli("Stage 6a: model card", [(("--repo",), dict(default="aimlcompanion/llm-training-lab")),
                                     (("--subfolder",), dict(default="lltCapstoneAssistant/dpo-merged")),
                                     (("--limitations",), dict(default=None, help="a text file with your own limitations section"))])
    R = {s: read_json(path(a, "records", s + ".json")) for s in ("data", "sft", "label", "dpo", "scores", "judge")}
    card, sft, lab, dpo, sc, jd = R["data"]["card"], R["sft"], R["label"], R["dpo"], R["scores"], R["judge"]
    jt = JUDGE[a.tier]
    comp = {k.split("|")[1]: v for k, v in jd["comparisons"].items() if k.startswith(jt + "|")}
    cal = jd["calibration"][jt]
    ife = sc["ifeval"]
    xs = sc["xstest"]
    import os
    ref = read_json(path(a, "records", "refusal.json"))["perModel"] if os.path.exists(path(a, "records", "refusal.json")) else {}

    def refcell(m, lbl):
        if m not in xs:
            return "n/a"
        return pct(xs[m][lbl + "Refusal"]) + (" / %s" % pct(ref[m][lbl]["judge"]) if m in ref else "")
    rows = chr(10).join("| %s | %s (%s) | %s | %s |" % (m, pct(ife[m]["promptStrict"]), ci(ife[m]["promptStrictCI"]),
                                                     refcell(m, "unsafe"), refcell(m, "safe"))
                     for m in ("base", "sft", "dpo", "official") if m in ife)
    wr = "\n".join("| %s | %s (%s) | %s (%s) | %s / %s tokens |" % (
        k, pct(v["winRate"]), ci(v["ci"]), pct(v["lengthControlled"]["lc"]), ci(v["lengthControlled"]["ci"]),
        v["meanTokens"][v["x"]], v["meanTokens"][v["y"]]) for k, v in comp.items())
    lim = open(a.limitations, encoding="utf-8").read() if a.limitations else "Write this section from your own evaluation."
    text = f"""---
license: apache-2.0
base_model: {BASE}
library_name: transformers
pipeline_tag: text-generation
tags: [capstone, sft, dpo, lora, llm-training-lab]
---

# lltCapstoneAssistant: an instruction-following assistant from {BASE}

The reference model of the LLM Training Lab capstone. It starts from the **base** model {BASE} and is trained in two
stages: supervised fine-tuning (SFT) on {card['sft']['n']:,} single-turn conversations, then direct preference optimization
(DPO) on {lab['n_pairs']:,} preference pairs sampled from the SFT model itself and labelled by the reward model {RM[a.tier]}.
Both stages train a LoRA adapter (rank {sft['config']['lora_r']}) that is merged into the weights here.

It is a teaching model, built to show the method and the evaluation. It is weaker than {OFFICIAL}, Qwen's own post-trained model
from the same base (table below). Do not use it where mistakes cost anything.

## Prompt format

Single turn only. The base model's chat tokens are untrained, so the model uses a plain-text template and ends every reply
with `<|endoftext|>`. `tokenizer.apply_chat_template` builds it for you.

```
{TEMPLATE}<your reply is generated here><|endoftext|>
```

## Training data

- SFT: {card['sft']['n']:,} conversations ({card['sft']['tokens']:,} tokens) sampled from allenai/Dolci-Instruct-SFT (ODC-BY; each
  source keeps its own licence, listed on that dataset's card). Mix by group: {', '.join(f'{k} {v:,}' for k, v in card['sft']['per_group'].items())}.
  Single-turn English only, no tool calls, replies that name a model or lab removed, and every prompt that shares a
  13-word span with an IFEval, Arena-Hard v2.0 or XSTest prompt removed ({card['sft']['drops'].get('overlaps an evaluation prompt', 0)} dropped).
  The WildChat replies in that set were written by GPT-4.1, and its Precise IF subset trains the same kinds of checkable
  constraints IFEval tests, so part of the IFEval gain reflects matched constraint types.
- DPO: {card['pref']['n']:,} prompts from HuggingFaceH4/ultrafeedback_binarized (MIT; prompts only), {a.cfg['k_samples']} replies
  each from the SFT model, scored by {RM[a.tier]}. Chosen = the best reply that ended on its own, rejected = the worst, kept when the
  score gap was at least {lab['minGap']}. Chosen replies were longer in {pct(lab['chosenLongerShare'])} of pairs.

## Evaluation

IFEval prompt-level strict accuracy (541 prompts, greedy, 95% bootstrap interval) and XSTest refusal rates on
200 unsafe and 250 safe prompts, first 64 tokens, as string match / LLM refusal judge (both checked against
XSTest's 2,250 human labels; the string rule misses refusals that lecture instead of apologising):

| Model | IFEval prompt strict | Refuses unsafe | Refuses safe (over-refusal) |
|---|---|---|---|
{rows}

Pairwise win rate on {comp[next(iter(comp))]['n']} Arena-Hard v2.0 prompts, judged by {jt} (open weights, run locally) in both
orders. Length-controlled = the win rate predicted at equal reply length (a one-term version of AlpacaEval's LC).

| Comparison | Win rate (95% CI) | Length-controlled | Mean reply length |
|---|---|---|---|
{wr}

The judge was checked first on {cal['n']} HelpSteer3 pairs with clear human preferences: it agreed with the human label in
{pct(cal['softAgreement'])} (95% CI {ci(cal['softCI'])}) and gave the same verdict in both orders on {pct(cal['consistentShare'])} of pairs.

## Limitations

{lim}

## How it was made

Code, runbook and every run file: https://github.com/genieincodebottle/aiml-companion/tree/main/courses/llm-training/labs/lltCapstoneAssistant
"""
    p = path(a, "dpo", "merged", "README.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    write_json(path(a, "records", "card.json"), {"stage": "card", "tier": a.tier, "smoke": a.smoke, "path": p, "chars": len(text)})
    print(text)


if __name__ == "__main__":
    main()
