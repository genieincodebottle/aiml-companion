# Ticket B: two replicas, one idle

Two identical replicas (Llama-3.1-8B bf16 on one NVIDIA A10 each) behind a router that
sends each request to a replica chosen from its prompt prefix, so prefix caching works.
Users report slow first tokens. The dashboard shows fleet-average GPU utilisation of 60 percent.

Scrapes for each replica in replica0/scrapes and replica1/scrapes, logs in logs_replica*.txt.
Write your diagnosis in debug_lab/diagnoses/b_hot_prefix_imbalance.md.
