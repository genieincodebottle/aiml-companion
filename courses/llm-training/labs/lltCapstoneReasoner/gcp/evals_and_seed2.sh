#!/bin/bash
# Evaluations of the reference runs (vLLM, n = 64), then a third seed of the ablation and its evaluation.
cd ~/capstoneP3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false LLTRL_STEPS=~/capstoneP3/lltRL_steps
st() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> chain.status; }
V=~/vllm-venv/bin/python
T=~/llt-venv/bin/python
st "evals start"
[ -f evals/base-0.6b.json ] || $V evaluate_vllm.py --out evals/base-0.6b.json --model Qwen/Qwen3-0.6B-Base --n 64 --max_new 256 --mem 0.8 > logs/eval_base06.log 2>&1
[ -f evals/agg-dr_grpo_scale-0_s1.json ] || $V evaluate_vllm.py --out evals/agg-grpo_scale-1_s0.json,evals/agg-grpo_scale-0_s0.json,evals/agg-dr_grpo_scale-1_s0.json,evals/agg-dr_grpo_scale-0_s0.json,evals/agg-grpo_scale-1_s1.json,evals/agg-grpo_scale-0_s1.json,evals/agg-dr_grpo_scale-1_s1.json,evals/agg-dr_grpo_scale-0_s1.json --adapter runs/agg-grpo_scale-1_s0/adapter-final,runs/agg-grpo_scale-0_s0/adapter-final,runs/agg-dr_grpo_scale-1_s0/adapter-final,runs/agg-dr_grpo_scale-0_s0/adapter-final,runs/agg-grpo_scale-1_s1/adapter-final,runs/agg-grpo_scale-0_s1/adapter-final,runs/agg-dr_grpo_scale-1_s1/adapter-final,runs/agg-dr_grpo_scale-0_s1/adapter-final --model Qwen/Qwen3-0.6B-Base --n 64 --max_new 256 --mem 0.8 > logs/eval_ablation.log 2>&1
[ -f evals/base-1.7b.json ] || $V evaluate_vllm.py --out evals/base-1.7b.json --model Qwen/Qwen3-1.7B-Base --n 64 --max_new 256 --mem 0.8 > logs/eval_base17.log 2>&1
[ -f evals/q17_agg-dr_grpo_scale-0_s0.json ] || $V evaluate_vllm.py --out evals/q17_agg-grpo_scale-1_s0.json,evals/q17_agg-dr_grpo_scale-0_s0.json --adapter runs/q17_agg-grpo_scale-1_s0/adapter-final,runs/q17_agg-dr_grpo_scale-0_s0/adapter-final --model Qwen/Qwen3-1.7B-Base --n 64 --max_new 256 --mem 0.8 > logs/eval_q17.log 2>&1
st "evals done: $(ls evals | grep -v vprobe | wc -l) files"
pids=()
for a in grpo dr_grpo; do for s in 1 0; do
  $T train.py --steps 100 --lr 2e-5 --max_new 256 --eval_every 25 --ckpt_every 5 --save_at 50 --out runs/agg-${a}_scale-${s}_s2 --agg $a --scale $s --seed 2 > logs/seed2_${a}_${s}.log 2>&1 &
  pids+=($!); sleep 15
done; done
for p in "${pids[@]}"; do wait $p; done
st "seed2 training done"
[ -f evals/agg-dr_grpo_scale-0_s2.json ] || $V evaluate_vllm.py --out evals/agg-grpo_scale-1_s2.json,evals/agg-grpo_scale-0_s2.json,evals/agg-dr_grpo_scale-1_s2.json,evals/agg-dr_grpo_scale-0_s2.json --adapter runs/agg-grpo_scale-1_s2/adapter-final,runs/agg-grpo_scale-0_s2/adapter-final,runs/agg-dr_grpo_scale-1_s2/adapter-final,runs/agg-dr_grpo_scale-0_s2/adapter-final --model Qwen/Qwen3-0.6B-Base --n 64 --max_new 256 --mem 0.8 > logs/eval_seed2.log 2>&1
st "ALL DONE: $(ls evals | grep -v vprobe | wc -l) eval files"
