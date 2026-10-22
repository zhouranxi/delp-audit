#!/usr/bin/env bash
# launch_pair.sh <jobsA> <jobsB> <done_marker>
# Two shard-groups run concurrently, each with its own GPU-0 and GPU-1 worker.
cd "$(dirname "$0")" || exit 1
A=${1:?}; B=${2:?}; MARK=${3:?marker}
LOG="${MARK}_a.log" bash run_grid.sh "$A" 2 > /dev/null 2>&1 &
pa=$!
LOG="${MARK}_b.log" bash run_grid.sh "$B" 2 > /dev/null 2>&1 &
pb=$!
wait $pa $pb
touch "$MARK"
