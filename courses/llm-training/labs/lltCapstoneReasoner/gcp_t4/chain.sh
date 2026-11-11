#!/bin/bash
# usage: chain.sh GPU WAIT_NAME next_agg next_scale next_seed
GPU=$1; W=$2
until grep -q ALLDONE /kaggle/working/nbout/$W.log 2>/dev/null; do sleep 15; done
/tmp/run_cell.sh $GPU $3 $4 $5
