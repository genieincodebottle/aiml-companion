# Ticket C: speculation made the afternoon worse

Llama-3.1-8B bf16 on one NVIDIA A10 with n-gram speculative decoding (4 draft tokens).
Quiet periods are faster than before speculation. At busy times, tokens per second and
inter-token latency are worse than they were with speculation off. The team suspects the
draft model quality (n-gram acceptance) dropped.

Scrapes every 30 s of engine time in scrapes/, engine log lines in logs.txt.
Write your diagnosis in debug_lab/diagnoses/c_speculation_collapse.md.
