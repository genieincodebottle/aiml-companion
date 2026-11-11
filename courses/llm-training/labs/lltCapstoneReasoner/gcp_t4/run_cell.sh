#!/bin/bash
# usage: run_cell.sh GPU AGG SCALE SEED
GPU=$1; AGG=$2; SCALE=$3; SEED=$4
NAME="agg-${AGG}_scale-${SCALE}_s${SEED}"
. $HOME/venv/bin/activate
cd /kaggle/working
mkdir -p nbout
python /tmp/mkvariant.py /tmp/lltCapstoneReasoner.ipynb nbout/nb_${NAME}.ipynb $GPU $AGG $SCALE $SEED
for pass in 1 2 3 4; do
  echo "pass $pass start $(date -Is)" >> nbout/${NAME}.log
  /usr/bin/time -v jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.record_timing=True \
     nbout/nb_${NAME}.ipynb --output executed_${NAME}_pass${pass}.ipynb >> nbout/${NAME}.log 2>&1
  echo "pass $pass end $(date -Is)" >> nbout/${NAME}.log
  [ -f runs/${NAME}/done.json ] && break
done
echo "ALLDONE $(date -Is)" >> nbout/${NAME}.log
