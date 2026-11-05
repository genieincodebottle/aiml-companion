# fix_audit.py - apply the batch 3 content-audit fixes (2026-11-05) to the page sources. Idempotent: skips pairs already applied.
import os

HERE = os.path.dirname(os.path.abspath(__file__))
Q = chr(92) + chr(39)          # backslash + quote, as written inside the .mjs single-quoted strings
QQ = chr(92) * 2 + chr(39)     # as written inside make_fill.py normal strings


def sub(fname, pairs):
    p = os.path.join(HERE, fname)
    t = open(p, encoding="utf-8").read()
    for a, b in pairs:
        if a in t:
            t = t.replace(a, b)
        elif b not in t:
            raise SystemExit(f"MISSING in {fname}: {a[:80]}")
    open(p, "w", encoding="utf-8").write(t)
    print("ok", fname)


sub("body_part1.mjs", [
    ("XXSo every parameter carries four fp32 numbers.' },",
     "So every parameter carries four fp32 numbers.' },\n                    { text: 'Parts 1 and 2 train on random token ids, with each batch as its own target, so the model only learns to copy its input. Their loss values check that the code runs. They measure nothing about language.' },"),
    ("['4 x 512', '2,048', '2.72', '2.75', '2.90', '-1.1%'],", "['4 x 512', '2,048', '3.70', '3.70', '3.83', '+0.1%'],"),
    ("made for the forward pass. The gradients are not allocated yet. The 6d is the final LayerNorm" + Q + "s input",
     "made for the forward pass. The gradients are not allocated yet.' },\n                    { text: 'The 6d is the final LayerNorm" + Q + "s input"),
    ("plus an attention term (Chowdhery et al. 2022, appendix B).' },",
     "plus an attention term (Chowdhery et al. 2022, appendix B).' },\n                    { text: 'That attention term counts the full s x s score matrix, the PaLM convention, although causal masking skips about half of that work.' },"),
    ("matrix multiply (run llt-gpu-smoke)", "matrix multiply (run lltMemoryThroughput-T2-gpu-smoke)"),
    ("Our two Kaggle sessions measured 14,399 and 12,627 tokens/s for the same cell, so expect a spread of about 14% between T4 machines.",
     "Two sessions on different Kaggle T4 machines measured 14,399 and 12,627 tokens/s for this cell (runs lltMemoryThroughput-T2-precision and -T2-timing-session1), so expect about 12% between machines."),
    ("{ text: 'bf16 keeps fp32" + Q + "s 8 exponent bits with 7 mantissa bits, so it needs no loss scaling (Kalamkar et al. 2019). On an Ampere or newer GPU, capability 8.0 and up, use bf16 autocast without a GradScaler.' },",
     "{ text: 'On an Ampere or newer GPU, capability 8.0 and up, use bf16 autocast without a GradScaler. dlTraining covers why bf16 needs no loss scaling.' },"),
    ("trained normally, 9.90 to 2.26 in 30 steps.", "trained normally, 9.90 to 2.27 in 30 steps."),
])

sub("body_part3.mjs", [
    ("{ text: 'z-loss made no measurable difference in 600 steps at 10M. It lowered the final mean log Z by 0.2 to 0.5 and left the losses within 0.14 of the baseline.' },",
     "{ text: 'We make no claim for z-loss from these runs. With one seed its losses landed within 0.14 of the baseline in either direction, and it lowered the final mean log Z by 0.2 to 0.5.' },"),
    ("{ text: 'The NaN is partly an fp16 limit. The 10M baseline at 3e-2 already had an infinite attention logit at step 150, an overflow above fp16" + Q + "s maximum of 65,504. The QK-norm runs at 3e-2 failed with attention logits near 11, so another tensor overflowed first, and we didn" + Q + "t locate which.' },",
     "{ text: 'For the 10M baseline at 3e-2, the probe read an infinite attention logit at step 150 while the validation loss was still finite, 5.16. We didn" + Q + "t find out why.' },\n                    { text: 'The QK-norm runs at 3e-2 failed with attention logits near 11, so their NaN started in another tensor, which we didn" + Q + "t locate. A rerun in fp32 or bf16 would show how much of the 3e-2 limit comes from fp16.' },"),
    ("text: 'LR sensitivity separates the variants more clearly as size grows. QK-norm lowered it by 0.12 at 10M and by 0.97 at 27M. Every run here is one seed, so read these as a first data point.' },",
     "text: 'QK-norm lowered LR sensitivity by 0.12 at 10M and by 0.97 at 27M. Almost all of the 27M gap comes from one run, the baseline at 1e-2, which diverged and counts as its step-0 loss. One seed per point, so the effect of size is possible, not shown.' },"),
])

sub("head.mjs", [
    ("objective: 'next-token loss, plus z-loss in one part 3 variant',",
     "objective: 'parts 1 and 2 use random tokens as their own targets, so their loss only checks the code. Part 3 minimizes next-token loss on TinyStories, plus z-loss in one variant',"),
])

sub("quiz.mjs", [
    ("o: ['About 1.0 GB, fp16 weights and gradients only', 'About 1.5 GB, at 12 bytes per parameter', 'About 2.0 GB, at 16 bytes per parameter', 'About 0.5 GB, the fp32 weights only'],",
     "o: ['About 0.5 GB, fp16 weights and gradients only', 'About 1.5 GB, at 12 bytes per parameter', 'About 2.0 GB, at 16 bytes per parameter', 'About 1.0 GB, fp32 weights and gradients only'],"),
    ("o: ['About the same, as both are 16-bit formats', 'Faster, as bf16 needs no loss scaling', 'It stops with an unsupported-dtype error', 'About 5 times slower than fp16'],",
     "o: ['About 1.0x, the same speed as fp16', 'About 1.4x faster, as bf16 needs no loss scaling', 'It stops with an unsupported-dtype error', 'About 5x slower than fp16, slower even than fp32'],"),
    ("'Both ranks drew the same batches',", "'Both ranks drew the same batches every step',"),
])

sub("tail.mjs", [
    ("logged 9.90 to 2.26 over 30 steps", "logged 9.90 to 2.27 over 30 steps"),
    ("cite: 'Micikevicius et al. 2018, Mixed Precision Training (ICLR 2018)',", "cite: 'Micikevicius et al. 2018, Mixed Precision Training',"),
    ("Experiences on Accelerating Data Parallel Training (VLDB 2020)',", "Experiences on Accelerating Data Parallel Training',"),
    ("and the run saw half the data it reports.',", "and the run saw half the data it reports. Its lower throughput is launch-to-launch variation, since the work per step is the same.',"),
    ("Neither reports the widening factor at the 10M to 30M scale of this lab.',",
     "Neither states a widening factor at 27M, so read Wortsman" + Q + "s small-model learning-rate curves before running.',"),
    ("report that qk-layernorm lowers LR sensitivity at every size they trained and", "report that qk-layernorm lowers LR sensitivity at every size they trained, small models included, and"),
])

# make_fill.py edits were applied directly to that file.
