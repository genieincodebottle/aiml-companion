---
license: apache-2.0
language:
- en
tags:
- education
- gpt
- tinystories
- llm-training
datasets:
- roneneldan/TinyStories
---

# LLM Training Lab checkpoints

Checkpoints and files for the LLM Training Lab track on [AI-ML Companion](https://aimlcompanion.ai). Each folder belongs to one lab page. Learners use them to "start from our checkpoint" when a full training run would take too long on free hardware.

## lltTinyStories

Used by the lab "Your First Real Pretraining Run".

| File | What it is |
|---|---|
| `step2000.pt` | Our seed-0 baseline GPT (11,474,240 parameters: width 320, 8 layers, 8 heads, tied output layer) after 2,000 of 3,000 training steps on TinyStories, trained on Kaggle T4 GPUs in fp16 with a GradScaler. A PyTorch dict with `model` (the state dict), `step` and `cfg`. |
| `tokenizer.json` | The byte-level BPE tokenizer trained in the same lab, needed to read and write text with this model. |

Load it the safe way, as the lab notebook does:

```python
import torch
ckpt = torch.load("step2000.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])   # model built from the lab's GPT code with ckpt["cfg"]
```

The lab then trains steps 2,000 to 3,000 on your own GPU session and compares your result with ours.

## Data and licences

- Training data: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0).
- These checkpoints: Apache-2.0.
- These are small teaching models. They write simple children's stories and are not meant for any other use.
