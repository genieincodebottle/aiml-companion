# Per-Class Accuracy Analysis

> Sample results from final model (93.4% overall accuracy).

| Class | Accuracy | Notes |
|---|---|---|
| airplane | 94.2% | Well-separated from other classes |
| automobile | 96.1% | High accuracy - distinct shape features |
| bird | 88.3% | Confused with airplane (flying objects) |
| cat | 85.7% | Hardest class - confused with dog |
| deer | 92.1% | Sometimes confused with horse |
| dog | 87.4% | Confused with cat (similar textures) |
| frog | 95.8% | Distinct color/shape profile |
| horse | 93.6% | Moderate difficulty |
| ship | 95.2% | Distinct shape + background (water) |
| truck | 94.8% | Similar to automobile but larger |

## Hardest Classes

1. **cat** (85.7%) - Most confused with dog. Similar fur textures and body shapes.
2. **dog** (87.4%) - Second hardest. The cat/dog distinction is a classic challenge.
3. **bird** (88.3%) - Sometimes classified as airplane (both appear against sky backgrounds).

## Improvement Ideas for Hardest Classes

- Focused augmentation on cat/dog examples
- Mixup or CutMix specifically between confusable pairs
- Larger model capacity (deeper ResNet blocks for stage 2-3)
