#!/bin/bash
# Evaluations (vLLM, n = 64). Runs beside the seed-2 training, so vLLM gets 30 percent of the GPU memory.
cd ~/capstoneP3
source ~/vllm-venv/bin/activate        # puts ninja on PATH; vLLM compiles kernels with it
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
st() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> chain.status; }
M=0.3
st "evals2 start"
[ -f evals/base-0.6b.json ] || python evaluate_vllm.py --out evals/base-0.6b.json --model Qwen/Qwen3-0.6B-Base --n 64 --max_new 256 --mem $M > logs/eval2_base06.log 2>&1
[ -f evals/agg-dr_grpo_scale-0_s1.json ] || python evaluate_vllm.py --out evals/agg-grpo_scale-1_s0.json,evals/agg-grpo_scale-0_s0.json,evals/agg-dr_grpo_scale-1_s0.json,evals/agg-dr_grpo_scale-0_s0.json,evals/agg-grpo_scale-1_s1.json,evals/agg-grpo_scale-0_s1.json,evals/agg-dr_grpo_scale-1_s1.json,evals/agg-dr_grpo_scale-0_s1.json --adapter runs/agg-grpo_scale-1_s0/adapter-final,runs/agg-grpo_scale-0_s0/adapter-final,runs/agg-dr_grpo_scale-1_s0/adapter-final,runs/agg-dr_grpo_scale-0_s0/adapter-final,runs/agg-grpo_scale-1_s1/adapter-final,runs/agg-grpo_scale-0_s1/adapter-final,runs/agg-dr_grpo_scale-1_s1/adapter-final,runs/agg-dr_grpo_scale-0_s1/adapter-final --model Qwen/Qwen3-0.6B-Base --n 64 --max_new 256 --mem $M > logs/eval2_ablation.log 2>&1
[ -f evals/base-1.7b.json ] || python evaluate_vllm.py --out evals/base-1.7b.json --model Qwen/Qwen3-1.7B-Base --n 64 --max_new 256 --mem $M > logs/eval2_base17.log 2>&1
[ -f evals/q17_agg-dr_grpo_scale-0_s0.json ] || python evaluate_vllm.py --out evals/q17_agg-grpo_scale-1_s0.json,evals/q17_agg-dr_grpo_scale-0_s0.json --adapter runs/q17_agg-grpo_scale-1_s0/adapter-final,runs/q17_agg-dr_grpo_scale-0_s0/adapter-final --model Qwen/Qwen3-1.7B-Base --n 64 --max_new 256 --mem $M > logs/eval2_q17.log 2>&1
st "evals2 first part: $(ls evals | grep -v vprobe | wc -l) files"
until grep -q "seed2 training done" chain.status; do sleep 60; done
[ -f evals/agg-dr_grpo_scale-0_s2.json ] || python evaluate_vllm.py --out evals/agg-grpo_scale-1_s2.json,evals/agg-grpo_scale-0_s2.json,evals/agg-dr_grpo_scale-1_s2.json,evals/agg-dr_grpo_scale-0_s2.json --adapter runs/agg-grpo_scale-1_s2/adapter-final,runs/agg-grpo_scale-0_s2/adapter-final,runs/agg-dr_grpo_scale-1_s2/adapter-final,runs/agg-dr_grpo_scale-0_s2/adapter-final --model Qwen/Qwen3-0.6B-Base --n 64 --max_new 256 --mem 0.8 > logs/eval2_seed2.log 2>&1
st "EVALS2 DONE: $(ls evals | grep -v vprobe | wc -l) eval files"
