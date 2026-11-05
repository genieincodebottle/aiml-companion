# make_fill.py - writes fill.json, the measured part 3 and timing texts that build_module.py substitutes.
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "..", "runs", "lltMemoryThroughput")
ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RUNS, "lltMemoryThroughput-T2-lr-*.json")))
assert len(ids) == 30, len(ids)
run_ids = "\n".join("            '%s'," % i for i in ids) + "\n            'lltMemoryThroughput-T2x2-dcp-repeat',\n            'lltMemoryThroughput-T2x2-dcp-control',\n            'lltMemoryThroughput-T2-timing-session1',\n            'lltMemoryThroughput-T2-gpu-smoke',\n            'lltMemoryThroughput-T2-notebook-timing',"

QUIZ_P3 = """        {
            q: 'At a learning rate of 1e-2, the 10M baseline\\'s largest attention logit reached 3,628 and its loss ended at 3.67. You add QK-norm and rerun the same seed. Predict the result.',
            o: ['Logits stay under 20, and the loss ends near 2.87', 'Logits stay under 20, yet the run still ends at NaN', 'Logits fall to about 1,800, and the loss to about 3.2', 'Nothing changes, as QK-norm matters only at 1B and up'],
            a: 0,
            e: 'QK-norm normalizes each head\\'s queries and keys, so their dot products grow only through learned gains. The largest attention logit was 15 and the loss 2.868. At 3e-2 every variant ended at NaN.',
        },
"""

IQA_P3 = """        {
            id: 'iv-lltmemorythroughput-4',
            difficulty: 'hard',
            level: 'Senior',
            tags: ['stability', 'learning rate', 'fp16'],
            question: 'You raise the learning rate of a small pretraining run and the loss goes to NaN. How do you find out why?',
            answer: `Separate a numeric overflow from a real divergence, then look at the two logit signals.

**Precision first**

In fp16 the loss can become NaN because one tensor passed 65,504, the fp16 maximum. A GradScaler then skips every step. In our runs it halved its scale 166 times, down to 0. Log the scaler's scale and the first step with a non-finite loss.

**The attention logits**

Log the largest query-key dot product on a fixed batch. In our 10M TinyStories baseline it rose from 22 at a learning rate of 1e-3 to 3,628 at 1e-2. QK-norm kept it under 20 at every learning rate.

**The output logits**

Log the mean log Z of the output softmax. If it drifts far from 0, z-loss at 1e-4 x (log Z)^2 targets it. In our 600-step runs z-loss lowered log Z by 0.2 to 0.5 and didn't change which learning rates trained. With one seed that is no claim either way.

**Then the usual suspects**

- Warm-up length. Wortsman et al. 2023 found longer warm-up lowers LR sensitivity.
- One bad batch. Rerun the step that spiked on its batch alone before blaming the optimizer.
- Gradient clipping. It can hide a real bug, so log the gradient norm before clipping.

**What our data shows**

QK-norm let the 27M model train at 1e-2, where the baseline diverged, and cut LR sensitivity from 4.019 to 3.048, almost all of that from the one diverged baseline run. Nothing trained at 3e-2 in fp16, so the next run I would do is the same sweep in fp32 or bf16, to see how much of that limit is fp16.`,
            keyPoints: [
                'Check the GradScaler and the first non-finite step to separate fp16 overflow from divergence',
                'Log the largest attention logit and the mean log Z on a fixed batch',
                'QK-norm kept attention logits under 20 and trained 27M at 1e-2 where the baseline diverged',
                'Rule out warm-up, a bad batch and clipping before blaming the optimizer',
            ],
        },
"""

fill = {
    "DCP_GAP_SENTENCE": "The two losses differ by 5.4 x 10⁻⁵. In a repeat run, two reloads of one checkpoint differed from each other by 2.1 x 10⁻⁵ and from the continued run by up to 3.8 x 10⁻⁵. So the gap is GPU run-to-run variation (run lltMemoryThroughput-T2x2-dcp-repeat). To check that this test can catch lost state, we also reloaded the weights alone with a fresh AdamW. After the same 2 steps its loss was 10.640458 against 10.722216 for the continued run, a gap of 0.08. A full reload landed within 2 x 10⁻⁵, and all 1,169 tensor shards on each rank matched their checksums (run lltMemoryThroughput-T2x2-dcp-control).",
    "KP_P3": "At a learning rate of 1e-2 the baseline\\'s largest attention logit reached 3,628 at 10M, and the 27M baseline diverged. QK-norm kept every attention logit under 20 and trained both sizes at 1e-2, but no variant trained at 3e-2 in fp16.",
    "P3_RUN_IDS": run_ids.strip(),
    "PAPER_WORTSMAN": "One seed, 600 steps, fp16. Attention logits grew with the learning rate in the baseline and stayed under 20 with QK-norm. QK-norm lowered LR sensitivity by 0.12 at 10M and 0.97 at 27M. We make no claim for z-loss from one seed. Almost all of the 27M gap comes from one baseline run at 1e-2 that diverged.",
    "PAPER_DEHGHANI": "At a learning rate of 1e-2 the baseline\\'s largest attention logit reached 3,628 at 10M and 7,900 at 27M. With QK-norm the largest in any run was under 20.",
    "PAPER_PALM": "MFU was 19.0% for GPT-2 small on one T4 and 13.3% for the 1.31B model on two. At 10M, z-loss lowered the final mean log Z by 0.2 to 0.5 and did not change which learning rates trained in 600 steps.",
    "OQ_RESULT": "First data point, one seed, fp16, 600 steps (runs lltMemoryThroughput-T2-lr-*). By the 0.1-nat rule QK-norm widened nothing at either size. Within 0.5 nats it moved the top of the range from 3e-3 to 1e-2 at both 10M and 27M, the same factor of 3.3. The grid steps by 3.3 there, so it can\\'t resolve a different factor. QK-norm cut LR sensitivity by 0.12 at 10M and 0.97 at 27M, and almost all of the 27M cut comes from one diverged baseline run.",
    "QUICK_MIN": "7",
    "QUICK_REACHES": "Part 1 took 407 s of GPU time on a Kaggle T4 in our run (run lltMemoryThroughput-T2-notebook-timing). It ends with the step 2 predictor checked against 17 measured peaks, all within 1.6%.",
    "T2_WALL": "about 21 min of GPU time for all three parts, 407 s, 500 s and 358 s, summed from our measured step times (run lltMemoryThroughput-T2-notebook-timing)",
    "QUIZ_P3": QUIZ_P3.rstrip("\n"),
    "IQA_P3": IQA_P3.rstrip("\n"),
}
with open(os.path.join(HERE, "fill.json"), "w", encoding="utf-8") as f:
    json.dump(fill, f, ensure_ascii=False, indent=1)
print("wrote fill.json", len(fill), "keys")
