# lltShip Try it spec: the merge and compression explorer (optional, class C)

Module 22, lltShip, class C (figures and Run Explorer; plan section 14). A widget is optional. If the lead builds one,
it goes in `lesson.tryIt` after step 5 and reads only `tryit_data.json` in this folder (built by `tryit_data.py` from
`runs/lltShip/lltShip-T3-panel-parents`, `-merge-grid`, `-gguf`, `-gptq`, `-qat`). No model runs in the browser.

## The one manipulation that teaches most

The learner picks a merge method and its knob and sees, on one plot, how much of each fine-tune's skill the merge
keeps. The point: adding two task vectors in full is not "both skills at once". With these two fine-tunes the Hindi
vector costs the instruction skill, and the methods trade one skill against the other. A second tab puts every
compressed version of the shipped merge on a quality-against-size plot.

## Data (all real, one A100 run, 2026-11-10)

`parents`: base, sft, hindi, each with hiBpb, enBpb, replyNll, endRate, meanTokens.
`merges[]`: name, method (task arithmetic, TIES, DARE), the same five numbers, and `kept` = [Hindi kept, replies kept],
where kept = (base - merge) / (base - parent) on that skill. `shipped`: the merge the selection rule picked.
`compressed[]`: GGUF types, GPTQ with two calibration sets, 3-bit RTN and QAT; name, MiB (where recorded), hiBpb,
enBpb, replyNll.

| Model | Method | Hindi bpb | Reply NLL | Hindi kept | Replies kept | Ends |
|---|---|---|---|---|---|---|
| base (parent) | none | 0.6131 | 1.3111 | | | 0.65 |
| sft (parent) | none | 0.6216 | 1.2448 | | | 0.86 |
| hindi (parent) | none | 0.6002 | 1.3694 | | | 0.64 |
| average, lambda 0.5 | task arithmetic | 0.5952 | 1.2732 | 1.386 | 0.571 | 0.82 |
| task arithmetic 1.0 | task arithmetic | 0.6037 | 1.3184 | 0.730 | -0.112 | 0.85 |
| TIES density 0.2 | TIES | 0.5963 | 1.3011 | 1.299 | 0.150 | 0.80 |
| task arithmetic 0.75 | task arithmetic | 0.5973 | 1.2889 | 1.226 | 0.335 | 0.85 |
| task arithmetic 1.25 | task arithmetic | 0.6133 | 1.3597 | -0.018 | -0.735 | 0.83 |
| task arithmetic 1.0 / 0.5, shipped | task arithmetic | 0.5986 | 1.2640 | 1.124 | 0.711 | 0.86 |
| task arithmetic 1.0 / 0.25 | task arithmetic | 0.6044 | 1.2496 | 0.674 | 0.928 | 0.83 |
| TIES density 0.5 | TIES | 0.5970 | 1.2984 | 1.244 | 0.191 | 0.85 |
| TIES density 1.0 | TIES | 0.5958 | 1.2959 | 1.341 | 0.228 | 0.82 |
| DARE p 0.5 | DARE | 0.6048 | 1.3262 | 0.645 | -0.229 | 0.90 |
| DARE p 0.9 | DARE | 0.6139 | 1.3690 | -0.058 | -0.875 | 0.82 |
| DARE p 0.9 seed 1 | DARE | 0.6141 | 1.3671 | -0.079 | -0.846 | 0.83 |

| Compressed version | MiB | Hindi bpb | Reply NLL | Ends |
|---|---|---|---|---|
| gguf f16 | 1142.7 | 0.5986 | 1.2640 | 0.86 |
| gguf Q8_0 | 609.8 | 0.5990 | 1.2647 | 0.84 |
| gguf Q4_K_M | 378.3 | 0.6297 | 1.3295 | 0.74 |
| gguf Q3_K_M | 331.0 | 0.7056 | 1.5147 | 0.61 |
| gguf Q2_K | 282.5 | 1.2250 | 2.8466 | 0.08 |
| gptq wikitext | 513.5 | 0.7176 | 1.4053 | 0.75 |
| gptq own | 513.5 | 0.6341 | 1.3686 | 0.73 |
| rtn 3-bit | not recorded | 0.9924 | 2.1921 | 0.23 |
| qat 3-bit | not recorded | 0.6880 | 1.5771 | 0.71 |

Shipped by the selection rule (largest smaller kept share): **task arithmetic 1.0 / 0.5**. With the default flags the notebook ships the equal average instead (lltShip-T3-notebook-defaults). Sizes for 3-bit RTN and QAT are not recorded; plot them on a separate 3-bit column, as lltShip.FIGURES.md says.

## Controls

1. Method picker, three rectangles (sentence case): "Task arithmetic", "TIES", "DARE".
2. One discrete slider for the picked method's knob, snapping only to recorded points: lambda (task arithmetic,
   including the two unequal settings as separate snap marks labelled "1.0 / 0.5" and "1.0 / 0.25"), density (TIES),
   p and seed (DARE). Keyboard arrows step one mark. 44 px touch targets.
3. Tab "Compressed versions" switches panel 1 to the size plot.

## Panels

**Panel 1, kept-share scatter.** x = Hindi kept, y = replies kept. Reference marks for the base (0, 0), the SFT model
and the Hindi model. Every recorded merge as a faint mark; the selected one solid with its label. A dashed box from
(0, 0) to (1, 1) labelled "keeps some of both".

**Panel 2, three bars** for the selected merge against the parents: Hindi bpb, English bpb, reply NLL (lower is
better), plus the stop rate as text.

**Compressed tab.** x = MiB (log scale), y = Hindi bpb or reply NLL (toggle). The fp16 merge as a dashed line.

## Must not claim

- That a method is better in general. One base, two fine-tunes, one seed per setting (two for DARE p 0.9), and the
  lambdas were tuned on the same items they are scored on.
- That a merge "has both skills" when one kept share is below 0.
- Any number not in `tryit_data.json`.

## Accessibility and theme

Method families by mark shape as well as colour; no hard-coded hex; dark and light; `aria-label` on the slider
("merge setting"); reduced motion: no animated transitions; 320 px to 1920 px with clamp() spacing.
