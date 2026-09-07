# Muse Glimmer Lab

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![GPU optional](https://img.shields.io/badge/GPU-optional-brightgreen.svg)](#what-runs-without-a-gpu)

A runnable companion to the post
[Meta's Muse Glimmer: An Agentic AI Model](https://aimlcompanion.ai/blog/meta-muse-glimmer-explained-2026).

Meta released [Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B)
on 10 August 2026, a 30B open-weight agentic model under Apache 2.0. At 4-bit it
is 16.8 GB of weights, so it runs on one 24 GB consumer GPU such as an RTX 4090,
or a Mac with 32 GB of unified memory. This lab takes the four design decisions
that make that possible and turns each one into something you can run and change.

| # | Experiment | What it shows | Without a GPU |
|---|---|---|---|
| 1 | [`01_hello.py`](scripts/01_hello.py) | Channel-scoped output, parsed into reasoning, tool call and answer | real parser, stand-in output |
| 2 | [`02_reasoning_strength.py`](scripts/02_reasoning_strength.py) | What the reasoning-strength knob actually costs | real parser, stand-in output |
| 3 | [`03_tool_loop.py`](scripts/03_tool_loop.py) | A full agentic loop over ATEM tool calls | real loop, stand-in output |
| 4 | [`04_kv_memory.py`](scripts/04_kv_memory.py) | Why the KV cache is 1.7 GiB instead of 104 GiB | **fully real**, no model needed |
| 5 | [`05_dflash_sim.py`](scripts/05_dflash_sim.py) | Why a diffusion drafter beats an autoregressive one | **fully real**, no model needed |

---

## Quickstart

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/llm/muse-glimmer-lab
uv sync
uv run python scripts/01_hello.py
```

If you do not have uv, `pip install uv`, or see
[the install guide](https://docs.astral.sh/uv/getting-started/installation/).
Plain `python scripts/01_hello.py` works too, since the offline path imports
nothing outside the standard library.

Run the tests with `uv run pytest`. There are 27 of them and they are worth
reading, because several encode a gotcha that costs an afternoon.

### What runs without a GPU

Be clear on what offline mode is. **It does not run Muse Glimmer.** The model
needs about 17 GB of VRAM and nothing here changes that.

What it does run, and what that is worth:

| Without a GPU | Status |
|---|---|
| Channel parsing, ATEM parsing, the agent loop (scripts 1-3) | Real code, exercised against a stand-in |
| KV-cache arithmetic (script 4) | **Real numbers**, derived from the published architecture |
| DFlash latency model (script 5) | **Real equation**, from the paper. Idealised, and the script says so |
| The 27 tests | Real, and several pin a documented gotcha |
| Muse Glimmer's actual output and judgement | Needs the hardware |

In offline mode `OfflineClient` emits genuine Glimmer-shaped output, channel
headers and ATEM blocks included, composed from your actual question and the
actual tool catalogue. The parsers cannot tell it apart from a served model, so
the formats and the loop are the real thing. What you lose is the model's
judgement. It will not surprise you, and it will not be wrong in the
interesting ways a real model is wrong. Point it at a real server as soon as
you have one.

---

## Running the real model

Glimmer needs about 17 GB for weights alone at 4-bit, plus a KV cache. Script 4
prints the arithmetic for your context length and card.

### llama.cpp, the 24 GB path

```bash
hf download meta-models/Muse-Glimmer-30B-GGUF --local-dir glimmer-gguf \
  --include "muse-glimmer-30B-kquant-17gb.gguf" \
  --include "mmproj-kquant.gguf" \
  --include "dflash-kquant.gguf"

./build/bin/llama-server \
  -m  glimmer-gguf/muse-glimmer-30B-kquant-17gb.gguf \
  --mmproj glimmer-gguf/mmproj-kquant.gguf \
  -md glimmer-gguf/dflash-kquant.gguf \
  -a muse-glimmer-30B \
  -ngl 99 -ngld 99 -c 131072 -np 4 \
  --host 127.0.0.1 --port 8080 --jinja \
  --temp 1.0 --top-p 0.95 --top-k 64
```

`-md` loads the DFlash drafter, and `-ngld 99` puts its layers on the GPU too.
Leaving `-ngld` off is a common way to make speculative decoding slower than no
speculative decoding at all, because the drafter then runs on the CPU and the
target model waits for it.

`-np 4` splits the context across four slots, so each one gets 32,768 tokens
rather than 131,072. Size `-c` for the number of slots you actually want.

Then point the lab at it:

```bash
GLIMMER_MODE=live GLIMMER_BASE_URL=http://127.0.0.1:8080/v1 \
  uv run python scripts/03_tool_loop.py
```

### Ollama, the one-line path

```bash
ollama run hf.co/meta-models/Muse-Glimmer-30B-GGUF
```

Convenient, and it hides the drafter and the vision projector, so you get
neither speculative decoding nor image input.

### vLLM, the serving path

```bash
vllm serve meta-models/Muse-Glimmer-30B \
  --max-model-len 131072 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto
```

Both parsers, always together. They key off the same channel framing, and the
reasoning parser is what forces `skip_special_tokens=False`. Without it the
channel markers are stripped before you see them and the output arrives as one
undifferentiated string, or empty.

---

## Four gotchas this lab encodes

**Never stop on `<|eom|>`.** It means end of message, not end of turn. Only
`<|eot|>` and `<|end_of_text|>` end a turn. Put `<|eom|>` in your stop list and
the model appears to request a tool and then die.
See [`channels.py`](glimmer/channels.py).

**Reasoning strength is set in two places and the template wins.** The model
card tells you to write `Reasoning strength: low` in the system prompt. The
chat template appends its own directive after yours and defaults to high, so
the system prompt alone may do nothing. See
[`client.system_prompt`](glimmer/client.py).

**ATEM arguments are all strings.** The format has no types, so `top_k` arrives
as `"3"`. Coerce against the tool schema before dispatch, or your integer
argument silently becomes a string. See
[`atem.coerce_arguments`](glimmer/atem.py).

**Weights are not the footprint.** The vision projector and the DFlash drafter
add 3 GB, and the KV cache adds more. Script 4 prints the row where a 24 GB
card stops fitting.

---

## Layout

```
glimmer/
  config.py       architecture and quant facts, each with its source
  channels.py     the <|start|> ... to=X ... <|message|> state machine
  atem.py         the XML tool-call format, parse and render
  tools.py        three tools, no network, no keys
  agent.py        the loop, about forty lines and no framework
  client.py       LiveClient (any OpenAI-compatible endpoint) and OfflineClient
  memory.py       KV-cache arithmetic with GQA and sliding window switchable
  speculative.py  DFlash latency model from the paper's equation (1)
scripts/          the five experiments
tests/            27 tests, several of which document a gotcha
```

---

## Sources

Every number in `config.py` carries a source comment. The primary ones are

- [Model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) for the architecture
- [Official GGUF repo](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) for quant names, sizes and llama.cpp flags
- [Meta AI research blog](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) for the training recipe and measured speedups
- [vLLM recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B) for the channel format and parser flags
- [Unsloth guide](https://unsloth.ai/docs/models/muse-glimmer) for community quants and fine-tuning
- [DFlash, arXiv:2602.06036](https://arxiv.org/abs/2602.06036) for the drafter. That paper is by Chen, Liang and Liu at UC San Diego, not by Meta. Glimmer adopts the technique.

MIT licensed. Muse Glimmer itself is Apache 2.0.
