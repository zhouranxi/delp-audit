#!/usr/bin/env bash
# launch_ladgain.sh -- detached driver for the joint (gain x lambda) ladder grid.
# Runs from dendritic-lab because ladder.py imports exp4/exp6 from there.
# Block mode: 21 (rung, T) blocks x 180 cells, packed onto 4 lanes over gpu0/gpu1.
cd /mnt/mydisk/xinguan-dendritic/dendritic-lab || exit 1
LOG=ladgain_b.log DEV="0 0 1 1" bash /mnt/mydisk/xinguan-dendritic/bench/run_lanes.sh \
    jobs_ladgain_batch.txt 4 "$HOME/venvs/dl/bin/python"
touch DONE_ladgain
