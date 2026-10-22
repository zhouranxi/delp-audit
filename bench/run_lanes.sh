#!/usr/bin/env bash
# run_lanes.sh -- N workers over M GPUs, where N may exceed M.
#   usage:  bash run_lanes.sh jobs.txt [nlanes] [python]
# jobs.txt (tab separated):  <lane> <tag> <args...>
# DEV is the lane -> device map, e.g. DEV="0 0 1 1" puts 2 workers on gpu0 and 2
# on gpu1. This box shares gpu2/gpu3 with other people's long jobs, so the
# default deliberately never touches them.
set -u
JOBS=${1:-jobs.txt}
NLANE=${2:-4}
PY=${3:-$HOME/venvs/dl/bin/python}
LOG=${LOG:-lanes.log}
DEV=${DEV:-"0 0 1 1"}
read -ra DEVS <<< "$DEV"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

: > "$LOG"
echo "[$(date +%H:%M:%S)] start lanes=$NLANE devs=$DEV jobs=$JOBS" >> "$LOG"

worker() {
  local lane=$1
  local dev=${DEVS[$lane]}
  local n=0
  while IFS=$'\t' read -r ln tag args; do
    [ -z "${ln:-}" ] && continue
    case "$ln" in \#*) continue;; esac
    [ "$ln" = "$lane" ] || continue
    # resume-safe: skip a cell whose output already exists and is non-empty
    out=$(printf '%s' "$args" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    if [ -n "$out" ] && [ -s "$out" ]; then continue; fi
    n=$((n+1))
    echo "[$(date +%H:%M:%S)] lane$lane/gpu$dev START $tag" >> "$LOG"
    if CUDA_VISIBLE_DEVICES=$dev $PY $args >> "$LOG" 2>&1; then
      echo "[$(date +%H:%M:%S)] lane$lane/gpu$dev DONE  $tag" >> "$LOG"
    else
      echo "[$(date +%H:%M:%S)] lane$lane/gpu$dev FAIL  $tag" >> "$LOG"
    fi
  done < "$JOBS"
  echo "[$(date +%H:%M:%S)] lane$lane finished $n jobs" >> "$LOG"
}

for l in $(seq 0 $((NLANE-1))); do
  worker "$l" &
done
wait
echo "ALL DONE $(date +%H:%M:%S)" >> "$LOG"
