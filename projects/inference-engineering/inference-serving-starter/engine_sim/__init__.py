"""engine_sim: a deterministic, physics-based model of an LLM serving engine.

This is the "customer" in this project. It is GIVEN. Do not rewrite it.
It is a model, not a GPU: it speaks the OpenAI-compatible HTTP API and exposes
vLLM-style Prometheus metrics, so the benchmark harness you write also works
against a real `vllm serve` (see gpu/).
"""
