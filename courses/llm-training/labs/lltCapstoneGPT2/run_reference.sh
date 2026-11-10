#!/usr/bin/env bash
# run_reference.sh - the H100 reference run for lltCapstoneGPT2, end to end. See RUNBOOK.md.
#
#   bash run_reference.sh calibrate   # about 20 min: prep 1 file, 40 steps at micro 32 and 64, gpt2 eval, projection
#   bash run_reference.sh full        # prep 14 files, train 19,073 steps, branch decays, evals, package
#
# Safe to rerun after any stop: data prep is skipped when done, training resumes from the newest checkpoint,
# finished branches and evals are skipped.
set -euo pipefail
MODE="${1:-full}"
W="${WORKDIR:-/workspace}"
MICRO="${MICRO:-32}"
PRICE="${PRICE_PER_HOUR:-2.69}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p "$W/run"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no GPU visible"
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert v >= (2, 4), "need torch >= 2.4, found " + torch.__version__
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0))
EOF
pip install -q "tiktoken==0.14.0" pyarrow huggingface_hub numpy
nvidia-smi > "$W/run/nvidia-smi.txt"
pip freeze > "$W/run/pip-freeze.txt"

TRAIN="torchrun --standalone --nproc_per_node=${NPROC:-1} train.py"

if [ "$MODE" = "calibrate" ]; then
  [ -f "$W/data_cal/index.json" ] || python prep_fineweb.py --out "$W/data_cal" --files 1 | tee "$W/run/calibrate_prep.log"
  for M in 32 64; do
    $TRAIN --data "$W/data_cal" --out "$W/cal$M" --run-id "cal-micro$M" --micro "$M" --max-steps 40 --warmup 10 \
      --stable-end 40 --save-every 40 --milestones 40 --export-every 40 --val-every 40 --sample-every 40 \
      2>&1 | tee "$W/run/calibrate_micro$M.log" || echo "micro $M failed (out of memory?); see the log"
  done
  pip install -q "lm-eval[hf]==0.4.13"
  python eval.py --baseline --out "$W/run/evals_calibration" 2>&1 | tee "$W/run/calibrate_eval.log"
  python calib_summary.py "$W" "$PRICE" | tee "$W/run/calibrate_summary.txt"
  rm -rf "$W"/cal32/ckpt "$W"/cal64/ckpt
  exit 0
fi

# ---- full ----
if [ ! -f "$W/data/index.json" ]; then
  python prep_fineweb.py --out "$W/data" 2>&1 | tee "$W/run/prep.log"
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--HuggingFaceFW--fineweb-edu"   # frees about 30 GB
fi

$TRAIN --data "$W/data" --out "$W/run" --micro "$MICRO" --resume auto 2>&1 | tee -a "$W/run/train.log"

# Branch decays from stable-phase checkpoints: one model per token budget, 5% decay, plus a 10% decay at 2.5B.
branch () {   # name stable_end max_steps
  local OUT="$W/branches/$1"
  [ -f "$OUT/done" ] && return 0
  local SRC="$W/run/ckpt/step_$(printf '%06d' "$2")"
  local RES="$SRC"
  [ -d "$OUT/ckpt" ] && [ -n "$(ls "$OUT/ckpt" 2>/dev/null | grep -v tmp || true)" ] && RES=auto
  $TRAIN --data "$W/data" --out "$OUT" --run-id "lltCapstoneGPT2-ref-124m-$1" --micro "$MICRO" --resume "$RES" \
    --stable-end "$2" --max-steps "$3" --milestones "$3" --save-every 100 --export-every 1000000 \
    --val-every 25 --sample-every 1000000 2>&1 | tee -a "$OUT.log"
  # keep only the branch's final checkpoint (1.5 GB each, est.)
  find "$OUT/ckpt" -maxdepth 1 -name 'step_*' ! -name "step_$(printf '%06d' "$3")" -exec rm -rf {} +
  touch "$OUT/done"
}
mkdir -p "$W/branches"
branch 0p5B-d5 906 954
branch 1B-d5 1813 1908
branch 2p5B-d5 4529 4769
branch 2p5B-d10 4292 4769
branch 5B-d5 9060 9537

# Evals: the final model and gpt2 on all 4 tasks, each budget model on all 4, then a HellaSwag curve.
pip install -q "lm-eval[hf]==0.4.13"
E="$W/run/evals"
[ -f "$E/lltCapstoneGPT2-eval-openai-gpt2.json" ] || \
  python eval.py --weights "$W/run/weights/weights_step_019073.pt" --baseline --val "$W/data/val.bin" --out "$E" 2>&1 | tee -a "$W/run/eval.log"
for B in 0p5B-d5 1B-d5 2p5B-d5 2p5B-d10 5B-d5; do
  [ -f "$E/$B.done" ] && continue
  python eval.py --weights "$W/branches/$B/weights/weights_step_*.pt" --out "$E/$B" 2>&1 | tee -a "$W/run/eval.log"
  touch "$E/$B.done"
done
[ -f "$E/curve.done" ] || { python eval.py --weights "$W/run/weights/weights_step_*.pt" --tasks hellaswag \
  --out "$E/curve" 2>&1 | tee -a "$W/run/eval.log"; touch "$E/curve.done"; }

# Package what we need back (small) and list what can be hosted (large).
cd "$W"
tar czf "$W/deliverables.tar.gz" run/*.json run/*.log run/*.txt run/evals branches/*.log \
  $(ls branches/*/*.json 2>/dev/null) data/index.json
du -sh run/ckpt/* run/weights branches/*/weights 2>/dev/null | tee "$W/run/sizes.txt"
echo "DONE. Send $W/deliverables.tar.gz. Checkpoints: $W/run/ckpt, weights: $W/run/weights, branches: $W/branches"
