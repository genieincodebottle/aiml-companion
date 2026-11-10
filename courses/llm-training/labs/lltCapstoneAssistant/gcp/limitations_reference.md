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
