#!/usr/bin/env bash
# launch_d2wm.sh -- D2 tau_a law test + write-gate arms.
# Runs from bench/ because d2_wm.py imports d2_comp_snn from here.
# DEV puts 2 workers on gpu2 and 2 on gpu3: gpu0/gpu1 are busy with the ladder
# grid, and these jobs need <1 GB, so they squeeze in beside other users.
cd /mnt/mydisk/xinguan-dendritic/bench || exit 1
LOG=d2wm.log DEV="2 2 3 3" bash run_lanes.sh jobs_d2wm.txt 4 "$HOME/venvs/dl/bin/python"
touch DONE_d2wm
