#!/bin/bash
# run_fashion_all.sh -- second-benchmark full pipeline on the server.
# Stage 1: lambda sweep (66 jobs) -> aggregate (val rule) -> stage 2: main table (80 jobs).
# Launch with:  nohup bash run_fashion_all.sh > logs_fashion/run.log 2>&1 &
set -u
cd /mnt/mydisk/xinguan-dendritic/dendritic-lab
PY=/mnt/mydisk/dendrite-mve/venv/bin/python
mkdir -p results_fashion logs_fashion
# 14 workers; GPUs 2/3 shared with the owner's Qwen3.8-27B (small MLP footprint)
GPUS=(0 0 0 0 1 1 1 1 2 2 3 3 0 1)
NW=${#GPUS[@]}

echo "[$(date +%H:%M:%S)] stage 1: lambda sweep, ${NW} workers"
for i in $(seq 0 $((NW-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY exp6_fixed_fashion.py --lam-rule val \
    --task-id $i --n-tasks $NW --stage sweep \
    --out results_fashion/sweep_$i.json > logs_fashion/sweep_$i.log 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] sweep finished"

$PY aggregate_fashion.py results_fashion/sweep_*.json > logs_fashion/aggregate.log 2>&1 \
  || { echo AGGREGATE_FAILED; exit 1; }
LAMMAP=$(cat lam_map_fashion.json)
echo "lam-map: $LAMMAP"

echo "[$(date +%H:%M:%S)] stage 2: main table"
for i in $(seq 0 $((NW-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY exp6_fixed_fashion.py --lam-rule val \
    --task-id $i --n-tasks $NW --stage main --lam-map "$LAMMAP" \
    --out results_fashion/main_$i.json > logs_fashion/main_$i.log 2>&1 &
done
wait

echo "[$(date +%H:%M:%S)] ALL DONE"
touch results_fashion/ALL_DONE
