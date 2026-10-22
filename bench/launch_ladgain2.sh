#!/usr/bin/env bash
# launch_ladgain2.sh -- gain-axis extension of the joint grid.
# Four lanes on gpu0/gpu1 (gpu2/gpu3 are shared with other users' jobs).
# Jobs write into the same results_ladgain dir; ladder.py skips existing
# cells, so the gain=16 overlap row is free and this is resumable.
cd /mnt/mydisk/xinguan-dendritic/dendritic-lab || exit 1
LOG=ladgain2.log DEV="0 1 0 1" bash /mnt/mydisk/xinguan-dendritic/bench/run_lanes.sh \
    jobs_ladgain2.txt 4 "$HOME/venvs/dl/bin/python"
touch DONE_ladgain2
