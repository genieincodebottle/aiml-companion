# Ticket A: chat latency cliff

Model: Llama-3.1-8B bf16 on one NVIDIA L4 (engine_sim). Traffic was light for two minutes,
then a campaign went out. Users now report chat replies that start after many seconds and
then stall mid-answer. GPU utilisation looks fine, nothing crashed, error rate is zero.

Scrapes every 30 s of engine time in scrapes/, engine log lines in logs.txt.
Write your diagnosis in debug_lab/diagnoses/a_kv_saturation.md.
