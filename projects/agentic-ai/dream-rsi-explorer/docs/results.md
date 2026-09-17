# Results

Every number here comes from a live run with `gemini-3.5-flash`, measured on
2026-09-17. A live model answers differently each time, so your numbers will
differ. The pattern is what to compare.

**Scope.** One task (26-circle packing), one model, a policy with eight knobs,
and one full loop checked on 3 fresh seeds per policy. That is enough to watch
the mechanism work. It is not a benchmark, and it doesn't reproduce the paper's
results.

The objective is the paper's replay score, measured on real runs.

```
value = best score - 0.004 * Gemini calls + 0.002 * calls per round
```

---

## 1. One search with the hand-written policy

```bash
uv run python run.py explore
```

```
policy          w2 a2 r1 p6 stop+0 s6 R6
rounds          6
Gemini calls    24  (24 valid layouts)
best score      2.5094   (best known for 26 circles is about 2.635)
best by round   2.500 2.500 2.509 2.509 2.509 2.509

Gemini usage: 24 calls, 11,455 input tokens, 9,175 output tokens, roughly $0.03 at list price
```

Took 38 seconds. The first layout already scored 2.50. The next 23 calls added 0.009.

## 2. Dream-RSI against the fixed loop

```bash
uv run python run.py compare
```

3 rounds, 4 policy versions per dreaming phase, 3 fresh seeds per policy.
Took 2 minutes 23 seconds.

```
Online rounds
round   fixed best  calls    dream best  calls   policy used by dream
0           2.5000     24        2.5000     24   w2 a2 r1 p6 stop+0 s6 R6
1           2.5131     24        2.4800      4   w1 a2 r1 p1 stop+0 s1 R3
2           2.4800     24        2.4800      4   w1 a2 r1 p1 stop+0 s1 R3
total       2.5131     72        2.5000     32

Fresh check, 3 real searches per policy on unseen seeds
              policy                              best   calls    value
hand-written  w2 a2 r1 p6 stop+0 s6 R6          2.5033    24.0   2.4153 +/- 0.0033
dream final   w1 a1 r1 p1 stop+0 s1 R3          2.4868     3.7   2.4752 +/- 0.0082

Replay predicted 2.4751 for the final policy. Real runs gave 2.4752.

Gemini usage: 175 calls, 80,717 input tokens, 59,820 output tokens, roughly $0.17 at list price
```

| Measure | Hand-written | Dream final |
|---|---|---|
| Gemini calls per search | 24 | **3.7** |
| Best score | 2.5033 | 2.4868 |
| Value | 2.4153 | **2.4752** |

**What happened.** Gemini's first layout for this task is already close to the
best it finds. The recorded trees showed most calls arriving after the last
improvement, and the policy-development agent said so in its reasons ("a high
share of calls after the last gain (0.62)"). Over three dreaming phases it cut
the rounds, the width and the patience. The final policy uses about 6.5 times
fewer calls and gives up 0.017 of best score.

**Replay was accurate here**, 2.4751 predicted against 2.4752 measured. Don't
expect that in general. The final policy only uses the first few attempts of
each search, which every recorded tree contains, so replay had full coverage.
A policy that searched wider than the recordings would get a less reliable
prediction.

## 3. Two findings from building it

**Let the model decide the arrangement, and let code do the arithmetic.** An
early version asked Gemini for complete layouts, centres and radii. Only 1 of 6
answers was valid, and every failure was an overlap between 0.02 and 0.16.
Asking for centres only, with radii fitted in code, gave 6 valid answers out of 6.

**This result depends on the model and the task.** A stronger first answer
leaves less for search to add, so the learned policy spends little. With a
weaker model, or a task where progress keeps coming, the same loop would learn to
keep searching. The objective decides the trade-off. Raise `--beta1` and calls
matter more; lower it and quality matters more.
