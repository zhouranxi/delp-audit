#!/bin/bash
# 用法: bash run_worker.sh <gpu_id> <joblist_file>
# 每行一个任务: "<model> <seed>"，顺序执行，结果写入 results/
set -u
GPU_ID=$1
JOBFILE=$2
cd /mnt/mydisk/dendrite-mve
export CUDA_VISIBLE_DEVICES=$GPU_ID
while read -r model seed; do
    [ -z "$model" ] && continue
    # 已有结果的任务跳过（支持断点续跑）
    if [ -f "results/${model}_seed${seed}.json" ]; then
        echo "[gpu$GPU_ID] skip ${model} seed${seed} (exists)"
        continue
    fi
    echo "[gpu$GPU_ID] start ${model} seed${seed} $(date +%H:%M:%S)"
    venv/bin/python train.py --model "$model" --seed "$seed" --device cuda --lr 1e-3 \
        --episodes 1500 --n-rules 4 --block-len 80 --episode-len 480 \
        >> "logs/${model}_seed${seed}.log" 2>&1 \
        || echo "[gpu$GPU_ID] FAILED ${model} seed${seed}"
done < "$JOBFILE"
echo "[gpu$GPU_ID] ALL_JOBS_DONE $(date +%H:%M:%S)"
