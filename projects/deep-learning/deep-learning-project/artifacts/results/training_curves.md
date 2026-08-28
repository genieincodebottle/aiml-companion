# Training Progression: 60% to 93%+

> Sample results from CIFAR-10 (60K images, 10 classes). Your results may vary.

## Experiment Progression

| # | Experiment | Train Acc | Test Acc | Delta | Key Change |
|---|---|---|---|---|---|
| 1 | baseline_cnn | 82.0% | 65.0% | -- | 3-layer CNN, Adam, no augmentation |
| 2 | add_augmentation | 85.0% | 75.0% | +10.0% | RandomCrop(32,4) + HorizontalFlip |
| 3 | resnet_skip | 95.0% | 85.0% | +10.0% | ResNet architecture + skip connections |
| 4 | cosine_lr | 97.0% | 91.0% | +6.0% | CosineAnnealingLR T_max=200 |
| 5 | mixed_precision | 96.5% | 92.0% | +1.0% | Mixed precision (torch.amp) |
| 6 | cutmix_final | 96.0% | 93.4% | +1.4% | CutMix + label smoothing |

## Key Hyperparameters (Final Configuration)

| Parameter | Value | Why |
|---|---|---|
| Optimizer | SGD + momentum 0.9 | Proven for vision tasks |
| Learning Rate | 0.1 | Found via LR finder |
| LR Schedule | CosineAnnealingLR (T_max=200) | Smooth decay to 0 |
| Weight Decay | 5e-4 | Standard regularization |
| Dropout | 0.2 | Prevents overfitting |
| Batch Size | 128 | GPU memory sweet spot |
| Epochs | 200 | With early stopping patience 20 |
| Label Smoothing | 0.1 | Reduces overconfidence |
