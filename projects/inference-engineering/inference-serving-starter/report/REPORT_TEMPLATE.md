<!--
Copy this file to report/REPORT.md and replace every <<...>> marker. `make check` fails while
any marker, "TODO" or template note is left. Keep the finished report to about one page of
text plus the tables and the plot (the test allows 350 to 1250 words).

The notes in italics marked "WEAK:" describe what a weak version of that section looks like.
Delete them from your report. They are here so you can read your own draft against them.
-->

# Serving <<model>> on <<GPU>>: benchmark, quantisation, speculation, GPU usage and fleet size

## 1. Setup

<<One short paragraph or a table: model and size, GPU and count, engine (engine_sim or vLLM version), dtype,
max_num_seqs, max_num_batched_tokens, prefix caching on/off, dataset (file, how it was made), SLO (TTFT and
TPOT limits and why those numbers), seed, date.>>

*WEAK: "I ran vLLM on a GPU." A reader cannot reproduce it, so they cannot trust it. Name every knob you did not change as well as the ones you did.*

## 2. Method

<<Closed-loop concurrency sweep for the throughput ceiling, and the open-loop Poisson run for the latency claim.
Warm-up excluded: how many requests or seconds, and why. Requests per level. Percentiles are over which population
(per request, or pooled gaps for ITL). How you checked your harness against the server's own metrics.>>

*WEAK: "I used the benchmark tool." Which loop, how many requests, what was thrown away, and what you compared it against?*

## 3. Results

<<Table, one row per concurrency level for the baseline (bf16): output tok/s, goodput req/s, SLO attainment, p50 and p99
TTFT, p99 ITL. Then the knee plot.>>

![throughput versus p99 latency](../results/knee_<<name>>.png)

Knee at concurrency <<N>>. Highest concurrency meeting the SLO at 95 percent attainment: <<M>>.

**Open loop versus closed loop.** <<results/open_bf16.json offers Poisson arrivals at 2.6 requests a second to the same server, which tops out near <<X>> requests a second in your sweep. Quote the open-loop p99 TTFT, its SLO attainment, and the p99 TTFT of the first and the last quarter of requests. Set them beside the closed-loop p99 TTFT at your highest level, and say why a closed loop could not show the growing queue.>>

*WEAK: only mean latency, or only one concurrency level, or a plot with no units. The mean hides the tail and one level hides the knee. For the open-loop paragraph: "the open loop was slower". Slower than what, by how much, and does the delay grow while the test runs?*

## 4. Quantisation comparison

<<bf16 vs fp8 vs int4 on the same GPU: tok/s at concurrency 1 and at the highest level, p99 TTFT, and the KV capacity each
left (tokens). Say WHICH resource each one relieved: weight bytes read per step (small batch) or KV room (large batch).
State the quality cost you did NOT measure.>>

*WEAK: "fp8 was faster." Faster at what concurrency, for which reason, and what did you give up? Quantisation is not free and this harness measures none of the quality side.*

## 5. Speculative decoding

<<ngram off vs on at LOW concurrency and at HIGH concurrency. Output tok/s for both, the ratio, and the mechanism: memory-bound
steps have spare compute so verifying k+1 tokens is nearly free; compute-bound steps pay for every extra token. State the
acceptance rate you assumed or measured, because the whole result depends on it.>>

*WEAK: "speculation gave 1.8x." At one concurrency, with one acceptance rate, stated as a law. The shape matters more than the number: it flips sign.*

## 6. GPU-hours per million tokens

<<GPU-hours per 1M tokens = GPUs x 1e6 / (tokens per second x 3600), using the output tok/s AT the SLO-safe operating
concurrency, not the peak. Result: <<gh>> GPU-hours per million output tokens at full utilisation, and <<eff>> effective GPU-hours
at your fleet's <<u>> utilisation at peak (the idle share is real usage).>>

<<Two SLOs, one sweep (python -m analysis.decide reads it under both). Say why the second limit exists and who would accept it.>>

| SLO | TTFT / TPOT limit | Operating concurrency | Output tok/s | GPU-hours per 1M output tokens | Replicas |
|---|---|---|---|---|---|
| <<chat>> | <<2 s / 0.15 s>> | <<c1>> | <<t1>> | <<gh1>> | <<r1>> |
| <<second>> | <<10 s / 0.3 s>> | <<c2>> | <<t2>> | <<gh2>> | <<r2>> |

<<Input tokens and the API. The mean request has <<P>> input and <<O>> output tokens, so one request uses <<req>> GPU-seconds at your operating point (GPUs / requests per second). Counting input and output tokens together that is <<blend>> GPU-hours per million tokens, which weighs an input token like an output token. The same request through a hosted API is worth <<api>> GPU-seconds of your GPU, using the API's rate for input and output tokens expressed as <<in>> and <<out>> GPU-hours per million tokens (how you derived those, and the date: <<basis>>). Self-hosting breaks even at <<bu>> utilisation: below it the API uses less GPU time per request, above it your fleet does. Say whether your fleet reaches it, and what the comparison leaves out.>>

*WEAK: an API rate copied from a blog post with no date, or throughput taken from the highest concurrency where the SLO was already broken. A second SLO row that equals the first. "Self-hosting is cheaper" with no break-even utilisation, or an API rate with no stated derivation.*

## 7. Fleet-size recommendation

<<Peak QPS <<q>> (where it came from), mean E2E <<s>> s at the operating point, Little's law gives <<L>> requests in flight.
One replica holds <<M>> within the SLO. With <<h>> percent headroom: <<R>> replicas, <<G>> GPUs, utilisation at peak <<u>>.
One sentence on what you would do about a 2x spike (queue, shed, autoscale on which metric).>>

*WEAK: "about 4 GPUs." With no peak, no in-flight arithmetic, and no headroom there is nothing to argue with and nothing to learn from.*

## 8. Measured on a real GPU

<<The GPU and provider, the vLLM version, the model, and the exact `vllm serve` command. Then a table for the bf16
baseline (fp16 on a T4, which has no bf16): output tok/s and p99 TTFT at concurrency 1 and at your highest level, engine_sim in one column and the real
run in the other. Then the cross-check: your harness and `vllm bench serve` at the same concurrency, both medians quoted,
and whether they agree within about 10 percent. One sentence for each number that moved, saying why.>>

| bf16 | engine_sim | <<GPU>>, vLLM <<version>> |
|---|---|---|
| tok/s at concurrency 1 | <<sim>> | <<measured>> |
| tok/s at concurrency <<N>> | <<sim>> | <<measured>> |
| p99 TTFT at concurrency <<N>> | <<sim>> | <<measured>> |

*WEAK: real numbers pasted with no cross-check, or a table with no explanation. The point is which simulated numbers held, which moved, and why.*

## 9. Threats to validity

- <<engine_sim is a model with documented simplifications, not a GPU. Which of your numbers did the real run move most, and which would you re-measure next on a bigger card?>>
- <<The dataset is synthetic lengths, not a real traffic log. How would a RAG or agent workload change the result?>>
- <<Single seed, <<n>> requests per level. How wide is the noise? Say how you know.>>
- <<Quality: nothing here measured accuracy after quantisation or the effect of speculation on output.>>
- <<Anything you assumed (acceptance rate, API rate, utilisation) and how sensitive the recommendation is to it.>>

*WEAK: "results may vary." Name the specific thing and the direction it would push the answer.*
