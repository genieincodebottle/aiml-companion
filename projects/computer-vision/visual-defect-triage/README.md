# Visual Defect Triage

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/module/computerVision/cvVitCapstone)**. Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.26-orange)
![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)

**Accuracy tells you how often the model is right. It tells you nothing about
when to trust it. This project builds the second thing.**

---

## 1. The problem

A factory line photographs every part. A model sorts them into six defect
classes plus pass, and a human reviews whatever the model is unsure about.

The interesting decision is not the classifier. It is the word "unsure". A
model that says 0.95 on a group of images it gets right 78 percent of the time
has made that word meaningless, and every routing rule built on it is
arbitrary.

## 2. One forward pass, three consumers

A Vision Transformer produces a 768 number embedding per image. Three separate
things read it, and none of them needs a second pass:

- the **classifier**, a linear layer over the embedding
- the **retrieval index**, so a reviewer sees the nearest labelled parts
- the **drift monitor**, which watches the embedding distribution move

Adding a second forward pass for retrieval is the obvious mistake and it
doubles the GPU bill for nothing.

## 3. Calibration is what makes a gate possible

Temperature scaling fits **one number** on validation data. Dividing every
logit by it cannot reorder them, so **accuracy is mathematically unchanged**
and only the confidence moves.

That is the whole point. It buys you a confidence you can put a threshold on
without touching the model's decisions.

```
fitted temperature: 1.626
accuracy 0.975 (unchanged)  |  ECE 0.013 -> 0.009
```

## 4. The result worth reading

`artifacts/slice_report.csv`, from a real run:

| slice | share | accuracy | ceiling |
|---|---|---|---|
| pass | 63.9% | 0.974 | **0.0167** |
| hairline_crack | 2.6% | **0.681** | 0.0083 |
| scratch | 13.4% | 1.000 | 0.0 |

`hairline_crack` is by far the worst class and offers **half the improvement
ceiling** of `pass`, because a ceiling is share multiplied by error rate and
`pass` carries 64 percent of the traffic.

Sorting slices by error rate sends you to the wrong one. The run asserts that
the ceilings sum to the error budget, so the report says where work is worth
doing rather than only where the model is weak.

## 5. Run it

### Clone

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/computer-vision/visual-defect-triage
```

### Set up with uv

[uv](https://docs.astral.sh/uv/) resolves and installs in seconds and keeps the
environment inside the project.

```bash
pip install uv

uv venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell or cmd

uv pip install -r requirements.txt
```

Plain `pip install -r requirements.txt` works identically. Python 3.10+.

**That install is small and CPU only.** Torch, timm and FAISS are deliberately
not in it, because the demo runs on synthetic embeddings and never loads a
network. Install `requirements-backbone.txt` on top only when you want to embed
real images.

### The argument, in order

```bash
python run.py data       # generate the synthetic embeddings   (~2s)
python run.py pipeline   # calibrate, gate, slice, retrieve    (~3s)
python run.py demo       # both of the above
python run.py slices     # the slice gate, as CI runs it
```

`run.py` needs **no install of this project**. It puts `src/` on the path
itself. There is a `Makefile` with the same targets if you prefer it, but
`make` is not on most Windows machines and `run.py` is.

### Run the tests

```bash
python run.py test       # or: pytest
```

29 tests, about 5 seconds. The ones that need torch skip themselves with
`pytest.importorskip`, so a clean install runs the whole suite green.

## 6. What each piece is for

| Piece | Why it exists |
|---|---|
| `src/calibrate.py` | The gate is a threshold on confidence, so the confidence has to be worth thresholding. One parameter, fitted on validation, cannot change a prediction. |
| `src/gate.py` | Routing is a business rule, not a model output. Two defect classes never auto-accept whatever the confidence, because a structural failure reaching a customer is not comparable to a scratch. |
| `src/metrics/slices.py` | Share multiplied by error rate. The number that says which slice is worth fixing. |
| `src/retrieval/` | The reviewer's real question is "have we seen this before", which a class label cannot answer and a neighbour can. |
| `src/review/capture.py` | Decisions become training data, tagged `mined` so they can never enter the evaluation set. |
| `src/monitor/drift.py` | The embedding is already computed, so watching its distribution is free. |

## 7. The notebook

`notebooks/visual_defect_triage_standalone.ipynb` is **standalone**. It
generates its own data, defines every function it uses, imports nothing from
`src/`, and writes nothing to disk.

**This is the recommended starting point if you are new to the topic.**

**Locally**, after the setup above:

```bash
jupyter lab notebooks/visual_defect_triage_standalone.ipynb
```

**On Google Colab**, no install at all:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/projects/computer-vision/visual-defect-triage/notebooks/visual_defect_triage_standalone.ipynb)

**On Kaggle**, published as a notebook you can copy and edit:

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/genieincodebottle/visual-defect-triage-vit)

`notebooks/_build_notebook.py` regenerates it. The notebook is authored as a
plain-text script so it reviews and diffs like code instead of JSON.

## 8. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `make: command not found` | no make on Windows | use `python run.py <cmd>` |
| `ModuleNotFoundError: src` | running a file directly from `src/` | use `python run.py`, it sets the path |
| `ModuleNotFoundError: timm` or `torch` | backbone extras not installed | expected. The demo does not need them, `pip install -r requirements-backbone.txt` if you want real images |
| `uv: command not found` after `pip install uv` | scripts dir not on PATH | `python -m uv venv`, or plain `pip` |
| slice test fails after you edit the generator | the baseline is a committed CSV | compare `artifacts/slice_report.csv` with `artifacts/slice_baseline.csv`, then update the baseline deliberately |
| numbers differ from this README | you changed a constant in `scripts/make_synthetic_data.py` | they are locked on purpose, see section 10 |

## 9. Layout

```
run.py                            zero-install entry point
configs/base.yaml                 backbone, thresholds, and the fitted temperature
scripts/make_synthetic_data.py    THE LOCKED GENERATOR, see section 10
scripts/check_slices.py           the CI gate
src/
├── config.py                     settings, read from configs/base.yaml
├── schemas.py                    the request and response contracts
├── data/{dataset,splits,transforms}.py   splits, and the leakage guard
├── models/{backbone,head,embed_cache,finetune}.py   torch lives HERE only
├── train.py                      the linear probe, numpy
├── calibrate.py                  temperature scaling, one parameter
├── gate.py                       accept / review / reject, plus the policy
├── metrics/{calibration,slices,gate_sim}.py   ECE, ceilings, review load
├── retrieval/{index,service}.py  nearest labelled neighbours
├── review/{capture,queue,agreement}.py  decisions as training data
├── monitor/drift.py              embedding distribution over time
└── run_pipeline.py               the whole argument end to end
api/                              FastAPI, with the batching seam
tests/                            29 tests
```

## 10. Track modules this covers

Each links straight to the module on AI-ML Companion.

| Module | Title |
|---|---|
| [`cvViT`](https://aimlcompanion.ai/module/computerVision/cvViT) | Vision Transformers |
| [`cvBackbones`](https://aimlcompanion.ai/module/computerVision/cvBackbones) | Choosing a Backbone |
| [`cvEval`](https://aimlcompanion.ai/module/computerVision/cvEval) | Evaluating Vision Systems |
| [`cvData`](https://aimlcompanion.ai/module/computerVision/cvData) | Images, Labels, and the Data You Actually Get |
| [`cvDeploy`](https://aimlcompanion.ai/module/computerVision/cvDeploy) | Vision in Production |
| [`cvIntro`](https://aimlcompanion.ai/module/computerVision/cvIntro) | What Computer Vision Actually Solves |

## 11. Honest limitations

**The embeddings are synthetic.** `scripts/make_synthetic_data.py` plants a
rare, subtle class and a labelling budget small enough to overfit, because
those two things are what make calibration and slice analysis worth doing. The
constants are locked, and four earlier settings were discarded: the first gave
accuracy 1.000 and a temperature of 0.12, which is a demo that cannot fail and
therefore teaches nothing.

**Four of the seven classes sit at accuracy 1.000.** Real defect data is not
that clean. The generator concentrates the difficulty into one class so the
ceiling argument is legible, which makes the slice table sharper than a real
one would be.

**Calibration is measured on the distribution it was fitted on.** A temperature
fitted in March is not automatically valid in September. `src/monitor/drift.py`
watches for the shift but nothing here refits automatically.

**The retrieval index is exhaustive.** Flat inner product over 40,000 vectors
is fine and an approximate index brings recall problems that are not the
subject of this project.

**The blog's figures and this repo's differ.** The walkthrough describes a
deployment at 96.2 percent overall and 71.4 on hairline crack. This repository
reports 97.5 and 68.1 because its data is synthetic. The shape of the argument
is identical and the numbers are not, and quietly aligning them would have
meant tuning the generator until it flattered the text.
