from transformers import GenerationConfig
PUSH_TO = None         # your own Hub repo, for example "your-name/qwen3-0.6b-hindi-instruct-merge"; needs an HF_TOKEN with write access

# 1. Generation settings that match how the model was trained: stop at <|endoftext|>, no 2,048-token default
GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, max_new_tokens=256, do_sample=False).save_pretrained("merged")

# 2. Prove the saved chat template writes exactly the prompt format the SFT model learned
reloaded = AutoTokenizer.from_pretrained("merged")
chat = [{"role": "user", "content": "Name a fruit."}]
print("template matches training format:", reloaded.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) == TEMPLATE.format("Name a fruit."))

# 3. The model card: what it is, what it was trained on, what we measured, and where it fails
m, b = runs[best], runs["base"]
CARD = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
tags: [merge, task-arithmetic, llm-training-lab]
datasets: [HuggingFaceH4/ultrafeedback_binarized, HuggingFaceFW/fineweb-2]
language: [en, hi]
---
# Qwen3-0.6B-Base, Hindi continued pretraining merged with instruction SFT

A weight merge ({best}) of two fine-tunes of Qwen/Qwen3-0.6B-Base, built in the LLM Training Lab:
an instruction SFT (LoRA, merged, 3,000 UltraFeedback chosen replies) and Hindi continued pretraining (320 steps, 5% English replay).

## Prompt format
Plain text, not ChatML. The base model's chat tokens were never trained, so the chat template writes:
"### Instruction:", the request, a blank line, "### Response:", and the reply ends with <|endoftext|>.

## Evaluation (held-out, paired against the base model)
| | Base | This model |
|---|---|---|
| Hindi bits per byte, 200 FineWeb-2 test texts | {b['hiBpb']:.4f} | {m['hiBpb']:.4f} |
| English bits per byte, 200 FineWeb-Edu texts | {b['enBpb']:.4f} | {m['enBpb']:.4f} |
| Reply NLL per token, 300 UltraFeedback test replies | {b['replyNll']:.4f} | {m['replyNll']:.4f} |
| Replies that stop on their own, 100 prompts, 256 tokens | {b.get('endRate', float('nan')):.2f} | {m.get('endRate', float('nan')):.2f} |

## Limits
- 0.6B parameters: replies are often wrong or repetitive. Not for factual, medical, legal or safety-relevant use.
- No safety tuning and no refusal training, and we did not measure refusals. Do not deploy it to the public.
- The Hindi gain is a small drop in bits per byte, not Hindi instruction following, which was never trained or measured.
- UltraFeedback labels come from GPT-4; OpenAI's terms on model outputs may apply to what you build with them.

## Licence
Apache-2.0, as the base model. UltraFeedback binarized is MIT; FineWeb-2 is ODC-By 1.0.
"""
open("merged/README.md", "w", encoding="utf-8").write(CARD)
print(sorted(os.listdir("merged")))

if PUSH_TO:
    from huggingface_hub import HfApi
    HfApi().create_repo(PUSH_TO, exist_ok=True)
    print(HfApi().upload_folder(folder_path="merged", repo_id=PUSH_TO))
