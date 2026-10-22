#!/bin/bash
# PR-003 用法: bash run_worker3.sh <gpu_id> <joblist_file>  （任务行: "<model> <seed>"）
set -u
GPU_ID=$1
JOBFILE=$2
cd /mnt/mydisk/dendrite-mve
export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p results_cue_adv
while read -r model seed; do
    [ -z "$model" ] && continue
    if [ -f "results_cue_adv/${model}_seed${seed}.json" ]; then
        echo "[gpu$GPU_ID] skip ${model} seed${seed} (exists)"
        continue
    fi
    echo "[gpu$GPU_ID] start ${model} seed${seed} $(date +%H:%M:%S)"
    venv/bin/python train2.py --model "$model" --seed "$seed" --device cuda --lr 1e-3 \
        --episodes 1500 --n-rules 4 --block-len 80 --episode-len 480 \
        --p-cue-valid 0.0 \
        --tag results_cue_adv \
        >> "logs/adv_${model}_seed${seed}.log" 2>&1 \
        || echo "[gpu$GPU_ID] FAILED ${model} seed${seed}"
done < "$JOBFILE"
echo "[gpu$GPU_ID] ALL_JOBS_DONE $(date +%H:%M:%S)"
