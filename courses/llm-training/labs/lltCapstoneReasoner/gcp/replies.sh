#!/bin/bash
# After the evaluations: full replies of the base model and five trained adapters, for the failure-case analysis.
cd ~/capstoneP3
source ~/vllm-venv/bin/activate
until grep -q "EVALS2 DONE" chain.status; do sleep 30; done
A=runs/agg-grpo_scale-1_s0/adapter-final,runs/agg-grpo_scale-1_s1/adapter-final,runs/agg-grpo_scale-0_s0/adapter-final,runs/agg-dr_grpo_scale-1_s0/adapter-final,runs/agg-dr_grpo_scale-0_s0/adapter-final
[ -f evals/replies-0.6b.json ] || python analyze_replies.py --out evals/replies-0.6b.json --adapters $A --mem 0.8 > logs/replies.log 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) REPLIES DONE" >> chain.status
