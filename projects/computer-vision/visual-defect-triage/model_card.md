# Model card - Visual Defect Triage

## What it is

A ViT-B/16 backbone with a linear classification head over frozen features, a
temperature-scaled confidence, and a three-way routing gate.

## Intended use

Teaching. It shows how a vision classifier becomes a production decision, and the
pieces around the model are the point rather than the model itself.

**Not for real safety-critical inspection without revalidation.** The thresholds
here were chosen against synthetic data.

## Data

The committed demo runs on synthetic embeddings from
`scripts/make_synthetic_data.py`, 12,000 images across 7 classes, deliberately
planted so that:

- `hairline_crack` is rare (3 percent) and sits close to `pass` in embedding
  space, because a hairline crack looks like a good part until you look properly
- only 3,000 of 8,400 training images carry a label, which is the labelling
  budget from the blog and also what makes the model overconfident

For real use, replace the generator with a manifest of your own images and run
the backbone. Nothing else changes.

## Metrics from the committed run

| metric | value |
| --- | --- |
| overall accuracy | 0.975 |
| hairline_crack accuracy | 0.681 |
| ECE before calibration | 0.013 |
| ECE after calibration | 0.009 |
| fitted temperature | 1.626 |
| auto-accept share | 79.6% |
| escaped errors | 6 of 1,433 accepted (0.42%) |

## Evaluation approach

Per-slice, always. Slices are ranked by improvement ceiling (share x error rate)
rather than by error rate, and the run asserts that the ceilings sum to the error
budget. Splits are grouped by production batch so near-duplicate photographs of
the same part cannot cross the boundary.

## Limitations

- Synthetic embeddings, so absolute numbers are illustrative. The structure they
  demonstrate is the deliverable, not the accuracy.
- The linear probe is calibrated by construction on well-specified data. The
  demo produces genuine overconfidence only because the labelling budget makes
  it overfit, which is realistic but is a property of this setup.
- Reviewer agreement is modelled, not measured. On a real line, measure it before
  asking a model to beat it.

## Ethical and operational notes

Two classes never auto-accept regardless of confidence. That rule is in
`src/gate.py` with a test, because it is the rule an optimisation removes while
trying to cut review volume.
