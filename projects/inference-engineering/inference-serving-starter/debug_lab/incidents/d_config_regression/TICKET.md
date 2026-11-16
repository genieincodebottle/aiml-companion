# Ticket D: p99 TTFT got worse after a config cleanup

Llama-3.1-8B bf16 on one NVIDIA A10, chat traffic. A cleanup changed the serve flags (see
serve_args.diff). The same traffic was replayed against the old and the new flags. After the
change, p99 time to first token is much worse and users see slow first tokens. Nothing crashed
and the error rate is zero. One engineer says the lower --gpu-memory-utilization caused it.

Scrapes every 30 s of engine time in before/scrapes and after/scrapes, logs in logs_before.txt
and logs_after.txt. Write your diagnosis in debug_lab/diagnoses/d_config_regression.md. This
incident is optional (not part of the five gates), but the checker will mark it if you write it.
