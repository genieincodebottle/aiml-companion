<!--
Copy this file to report/REPORT.md and replace every <<...>> marker. `make check` fails while
any marker, "TODO" or template note is left. Keep the finished report to about one page of
text plus the table and the plot (the test allows 350 to 1100 words).

The notes in italics marked "WEAK:" describe what a weak version of that section looks like.
Delete them from your report. They are here so you can read your own draft against them.
-->

# Serving <<model>> on <<GPU>>: benchmark, quantisation, speculation, cost and fleet size

## 1. Setup

<<One short paragraph or a table: model and size, GPU and count, engine (engine_sim or vLLM version), dtype,
max_num_seqs, max_num_batched_tokens, prefix caching on/off, dataset (file, how it was made), SLO (TTFT and
TPOT limits and why those numbers), seed, date.>>

*WEAK: "I ran vLLM on a GPU." A reader cannot reproduce it, so they cannot trust it. Name every knob you did not change as well as the ones you did.*

## 2. Method

<<Closed-loop concurrency sweep for the throughput ceiling, open-loop Poisson for the latency claim (if you ran one).
Warm-up excluded: how many requests or seconds, and why. Requests per level. Percentiles are over which population
(per request, or pooled gaps for ITL). How you checked your harness against the server's own metrics.>>

*WEAK: "I used the benchmark tool." Which loop, how many requests, what was thrown away, and what you compared it against?*

## 3. Results

<<Table, one row per concurrency level for the baseline (bf16): output tok/s, goodput req/s, SLO attainment, p50 and p99
TTFT, p99 ITL. Then the knee plot.>>

![throughput versus p99 latency](../results/knee_<<name>>.png)

Knee at concurrency <<N>>. Highest concurrency meeting the SLO at 95 percent attainment: <<M>>.

*WEAK: only mean latency, or only one concurrency level, or a plot with no units. The mean hides the tail and one level hides the knee.*

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

## 6. Cost per million tokens

<<Cost = GPU $/hour x GPUs / (tokens per second x 3600) x 1e6, using the output tok/s AT the SLO-safe operating
concurrency, not the peak. GPU price: $<<price>>/hour from <<source and date>>. Result: $<<cost>> per million output tokens.>>

*WEAK: a price copied from a blog post with no date, or throughput taken from the highest concurrency where the SLO was already broken.*

## 7. Fleet-size recommendation

<<Peak QPS <<q>> (where it came from), mean E2E <<s>> s at the operating point, Little's law gives <<L>> requests in flight.
One replica holds <<M>> within the SLO. With <<h>> percent headroom: <<R>> replicas, <<G>> GPUs, utilisation at peak <<u>>.
One sentence on what you would do about a 2x spike (queue, shed, autoscale on which metric).>>

*WEAK: "about 4 GPUs." With no peak, no in-flight arithmetic, and no headroom there is nothing to argue with and nothing to learn from.*

## 8. Threats to validity

- <<engine_sim is a model with documented simplifications, not a GPU. Which of your numbers would you re-measure first on a real card?>>
- <<The dataset is synthetic lengths, not a real traffic log. How would a RAG or agent workload change the result?>>
- <<Single seed, <<n>> requests per level. How wide is the noise? Say how you know.>>
- <<Quality: nothing here measured accuracy after quantisation or the effect of speculation on output.>>
- <<Anything you assumed (acceptance rate, price, utilisation) and how sensitive the recommendation is to it.>>

*WEAK: "results may vary." Name the specific thing and the direction it would push the answer.*
