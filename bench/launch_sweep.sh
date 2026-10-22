#!/usr/bin/env bash
# launch_sweep.sh -- run the two sweep shard-groups concurrently, wait, mark done.
cd "$(dirname "$0")" || exit 1
LOG=lad_sw_a.log bash run_grid.sh jobs_lad_sw_a.txt 2 > /dev/null 2>&1 &
pa=$!
LOG=lad_sw_b.log bash run_grid.sh jobs_lad_sw_b.txt 2 > /dev/null 2>&1 &
pb=$!
wait $pa $pb
touch SWEEP_LAD_DONE
