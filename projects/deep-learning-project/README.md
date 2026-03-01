# From Pixels to Production: CIFAR-10 Progressive Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen)

Iteratively optimize a CIFAR-10 image classifier from 60% baseline to 93%+ accuracy, documenting every experiment with a full diagnostics toolkit.

## Architecture

```
CIFAR-10 (60K images, 10 classes)
    |
    v
+---------------------------------------+
|   Progressive Training Pipeline        |
|                                        |
|   Exp 1: Simple CNN (baseline ~60%)   |
|   Exp 2: + Data Augmentation          |
|   Exp 3: ResNet-18 architecture       |
|   Exp 4: CosineAnnealing LR          |
|   Exp 5: Mixed Precision Training     |
|   Exp 6: Full optimization (93%+)    |
+---------------------------------------+
    |
    v
+---------------------------------------+
|   Diagnostics Toolkit                  |
|   LR Finder | Gradient Monitor        |
|   Per-class accuracy | Loss curves    |
+---------------------------------------+
```

## Problem Statement

Deep learning is an iterative process. This project demonstrates systematic experimentation: starting with a simple CNN baseline, progressively adding techniques (augmentation, skip connections, learning rate scheduling, mixed precision), and documenting the impact of each change.

## Approach

| # | Experiment | Key Change | Test Acc |
|---|---|---|---|
| 1 | baseline_cnn | 3-layer CNN, Adam | 65.0% |
| 2 | add_augmentation | RandomCrop + HorizontalFlip | 75.0% |
| 3 | resnet_skip | Skip connections + SGD | 85.0% |
| 4 | cosine_lr | CosineAnnealingLR | 91.0% |
| 5 | mixed_precision | torch.amp GradScaler | 92.0% |
| 6 | cutmix_final | CutMix + label smoothing | **93.4%** |

> Sample results. Your results may vary slightly due to random seed differences.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/deep-learning-project
pip install uv
uv venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
uv pip install -r requirements.txt

# Using Make (recommended)
make train                # Train the full progressive pipeline
make diagnostics          # Run diagnostics toolkit demo
make test                 # Run tests
make all                  # Full pipeline: test → train → diagnostics

# Or run directly
python -m src.train
python -m src.diagnostics
pytest tests/ -v

# Or use the pipeline script
bash scripts/run_training.sh
```

## Project Structure

```
deep-learning-project/
├── configs/
│   └── base.yaml                   # LR, batch size, epochs, augmentation flags per experiment
├── notebooks/
│   └── CIFAR10_Progressive.ipynb   # Complete walkthrough
├── src/
│   ├── __init__.py
│   ├── train.py                    # Progressive CIFAR-10 classifier (6 experiments)
│   └── diagnostics.py              # LR finder, gradient monitor, training tracker
├── tests/
│   ├── __init__.py
│   └── test_model.py               # pytest: ResBlock, CIFAR10Net, diagnostics tools
├── artifacts/
│   ├── checkpoints/                # Model checkpoints (.pt files)
│   └── results/
│       ├── training_curves.md      # Documented progression from 60% to 93%+
│       └── per_class_accuracy.md   # Which CIFAR-10 classes are hardest
├── docs/
│   └── experiment_log.csv          # Pre-filled experiment progression
├── scripts/
│   └── run_training.sh             # One-command: full 6-experiment pipeline
├── .gitignore
├── Makefile                        # make train | make test | make diagnostics
├── requirements.txt
└── README.md
```

## Key Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Optimizer | SGD + momentum 0.9 | Proven recipe for vision tasks |
| Learning Rate | 0.1 | Found via LR finder (Smith 2017) |
| LR Schedule | CosineAnnealingLR (T_max=200) | Smooth decay, no manual tuning |
| Weight Decay | 5e-4 | Standard regularization for SGD |
| Dropout | 0.2 | After global average pooling |
| Batch Size | 128 | GPU memory sweet spot |
| Label Smoothing | 0.1 | Reduces overconfidence |

## Interview Guide: How to Talk About This Project

### "Walk me through this project."

"I built a CIFAR-10 image classifier that progressively improved from 60% to 93%+ accuracy across 6 documented experiments. The key was systematic iteration: each experiment changed exactly one thing, so I could measure the impact. I also built a diagnostics toolkit with LR finder, gradient monitoring, and automated experiment logging."

### "What was the hardest part?"

"Getting past the 90% plateau. The jump from 85% to 91% came from CosineAnnealingLR -- constant LR was causing the model to oscillate around the optimum. The last 2% came from CutMix and label smoothing, which improved generalization on the hardest classes (cat/dog confusion)."

### "What would you do differently?"

"Three things: (1) Use a pretrained ResNet-18 with fine-tuning -- transfer learning would likely reach 95%+ faster. (2) Add Mixup in addition to CutMix for better regularization. (3) Use wandb instead of CSV for experiment tracking -- better visualization and hyperparameter sweep support."

### "How does this scale to production?"

"The model is 2.8M parameters and runs inference in <5ms on a GPU. For production: export to ONNX for faster inference, add a FastAPI wrapper, containerize with Docker, and add monitoring for prediction distribution drift. The experiment log format would integrate with MLflow for team collaboration."

### "Explain skip connections to a non-technical person."

"Imagine learning a complex recipe. Without skip connections, you must learn every step from scratch. With skip connections, you can say 'start with what you already know and just learn the adjustments.' This makes it much easier for the network to learn -- instead of learning the full transformation, it only learns the residual (the difference)."
