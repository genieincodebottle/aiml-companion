#!/usr/bin/env bash
# Re-measure every lltGeneration T1 figure in the pinned T1 environment (.venv-latest), one job at a time.
# Logs go to out/logs/. Timings are NOT taken here (measure_timing.py, idle machine).
set -e
cd "$(dirname "$0")"
PY=../../.venv-latest/Scripts/python.exe
mkdir -p out/logs
$PY -u train_small.py --seed 0 --threads 4 > out/logs/train.log 2>&1
$PY -u measure.py --threads 4 > out/logs/measure.log 2>&1
$PY -u run_steps.py > out/logs/steps.log 2>&1
$PY -u induction.py --seed 0 --threads 4 --steps 1500 --eval-every 25 > out/logs/toy_L2.log 2>&1
$PY -u induction.py --seed 0 --threads 4 --steps 1500 --eval-every 25 --layers 1 > out/logs/toy_L1.log 2>&1
$PY -u induction.py --seed 0 --threads 4 --steps 1000 --eval-every 25 --layers 1 --fixed-period > out/logs/toy_L1_fixed.log 2>&1
$PY -u run_research_cell.py --threads 4 > out/logs/research_cell.log 2>&1
$PY -u ts_induction.py --threads 4 > out/logs/ts_induction.log 2>&1
$PY -u ts_repeat_text.py --threads 4 > out/logs/ts_repeat.log 2>&1
$PY -u ts_sweep_induction.py --threads 4 > out/logs/ts_sweep.log 2>&1
echo ALL DONE
