# IPL 2026 Predictions — Reference Archive

A record of the **hand-curated predictions** posted to the live site for every IPL 2026 match. This folder is intended as a **reference**, not as runnable code.

## Want to actually run predictions?

Use [`scripts/predict_all_2026.py`](../scripts/predict_all_2026.py) in the project root. It loads the real scikit-learn ensemble (RF + XGB + GB + LR) from [`src/models.py`](../src/models.py), trains on pre-2026 historical IPL data, and predicts every 2026 match in chronological order with reproducible output.

```bash
cd ipl-match-predictor
pip install -r requirements.txt
python scripts/predict_all_2026.py
```

Expected runtime: 30-90 seconds. Outputs a per-match prediction table plus final accuracy.

## What's in here

```
predictions-2026/
├── README.md           ← orientation (you are here)
├── INDEX.md            ← per-match table: hand-curated pick vs actual result
└── python-scripts/     ← 66 self-contained exploratory .py prototypes (M1-M40 era)
```

## Headline numbers (hand-curated, with expert overlay)

- **72 matches** predicted (70 league + 2 playoff so far)
- **70 predictions resolved**, 2 upcoming
- **43 / 70 correct = 61.4% accuracy**

See [INDEX.md](INDEX.md) for the per-match table.

## Why two artifacts (archive + runnable pipeline)?

**The archive captures expert reasoning.** Each prediction in this folder was authored before the match with hand-tuned contextual adjustments (player matchups, dew expectation, momentum breaks). The reasoning narratives are pre-game and unedited — including the 27 wrong calls. That's a rare dataset of pre-match ML reasoning that wasn't retconned after the fact.

**The runnable pipeline ([`scripts/predict_all_2026.py`](../scripts/predict_all_2026.py)) is the deterministic baseline.** Pure ensemble output, no expert overlay. Useful for:
- Reproducing the result from your own machine
- A/B testing feature changes (modify `src/features.py`, rerun)
- Understanding what the model "really thought" before any expert adjustment

### How they compare on this season

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Random coin flip | 50.0% | Baseline |
| Hand-curated archive (with expert overlay) | **43 / 70 = 61.4%** | What was actually posted to the live site |
| Runnable trained ensemble | **34 / 70 = 48.6%** | Real reproducible number from `python scripts/predict_all_2026.py` |

**Honest result:** The trained ensemble underperforms a coin flip. The hand-curated expert overlay beats it by ~13 percentage points. The trained 4-model ensemble — even with Elo + momentum + H2H + venue features — is essentially random on this season.

Earlier drafts of this doc claimed ~65.7% for the runnable pipeline. That number was wrong — it came from a bug in `src.models.predict_match()` (17-feature dict vs 46-feature trained models) that silently fell back to pure-Elo expectation computed with target-row leakage. The 48.6% above is the actual ensemble output using `ensemble_predict_proba()` directly. See [`docs/ALGORITHM_WALKTHROUGH.md`](../docs/ALGORITHM_WALKTHROUGH.md) for the bug post-mortem.

**The headroom is real.** A learner who adds even a single well-chosen feature (dew, captain change, opening-pace matchup, day vs night) has a fair chance of beating both the 48.6% baseline and the 61.4% expert overlay.

## Learning paths through the archive

### Path 1 — Feature engineering for ranking problems
Pick 5 matches from [INDEX.md](INDEX.md). For each, open the Python prototype and study how the same 5 feature families (Elo, H2H, form, venue, momentum) get weighted differently based on context.

### Path 2 — Post-mortem on the 27 misses
The 27 wrong picks are where the real learning is. Open a wrong prediction's prototype and ask: which assumption broke? Was it a player-form misread? A venue model that didn't generalize to playoffs? Toss flipping the chase advantage?

### Path 3 — Beat the baseline
Run [`scripts/predict_all_2026.py`](../scripts/predict_all_2026.py) to get the raw ensemble baseline. Then modify `src/features.py` to add a feature you think matters (powerplay strike rate, captain change, weather). Rerun. Did accuracy go up?

### Path 4 — Build your own contextual overlay
Read 5 archive predictions and identify the "contextual adjustment" pattern (e.g., "+2% for home advantage if avg 1st innings > 180"). Codify it as a rule in Python. Apply on top of the runnable pipeline's output. Can you reproduce the +3 percentage point human edge?

## Important caveats

- **Reasoning is hand-curated.** The 4-model ensemble scores quoted in archive predictions (RF/XGB/GB/LR percentages) are realistic but representative — not output from a pickled scikit-learn model. The actual feature derivation, narrative reasoning, and contextual adjustments were authored at prediction time with reference to season form data.
- **The Python `.py` prototypes** vary in completeness. Some load real models from `../src/`, others have hand-tuned inline coefficients. Treat each as a snapshot, not a production pipeline.
- **For reproducible model output**, always use [`scripts/predict_all_2026.py`](../scripts/predict_all_2026.py), not the archive prototypes.

## Related project files

- [`../src/`](../src/) — the actual scikit-learn pipeline (data_loader, features, models, evaluate, predict)
- [`../scripts/predict_all_2026.py`](../scripts/predict_all_2026.py) — runnable 2026 predictor
- [`../data/raw/matches_2026.csv`](../data/raw/matches_2026.csv) — 2026 season data
- [`../notebooks/`](../notebooks/) — exploratory Jupyter notebooks
- [`../docs/`](../docs/) — methodology writeups

## License & use

All scripts in this folder are released for **educational use**. Reasoning text, feature taxonomies, and ensemble architecture explanations may be quoted with attribution.
