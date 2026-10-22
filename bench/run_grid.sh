#!/usr/bin/env bash
# run_grid.sh -- one worker per GPU, each draining its own queue.
#   usage:  bash run_grid.sh jobs.txt [ngpu] [python]
# jobs.txt format (tab separated):  <gpu> <tag> <args...>
set -u
JOBS=${1:-jobs.txt}
NGPU=${2:-4}
PY=${3:-$HOME/venvs/dl/bin/python}
LOG=${LOG:-run.log}

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

: > "$LOG"

worker() {
  local g=$1
  local n=0
  while IFS=$'\t' read -r gpu tag args; do
    [ -z "${gpu:-}" ] && continue
    case "$gpu" in \#*) continue;; esac
    [ "$gpu" = "$g" ] || continue
    # resume-safe: skip a run whose output already exists and is non-empty
    out=$(printf '%s' "$args" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    if [ -n "$out" ] && [ -s "$out" ]; then continue; fi
    n=$((n+1))
    echo "[$(date +%H:%M:%S)] gpu$g START $tag" >> "$LOG"
    if CUDA_VISIBLE_DEVICES=$g $PY $args >> "$LOG" 2>&1; then
      echo "[$(date +%H:%M:%S)] gpu$g DONE  $tag" >> "$LOG"
    else
      echo "[$(date +%H:%M:%S)] gpu$g FAIL  $tag" >> "$LOG"
    fi
  done < "$JOBS"
  echo "[$(date +%H:%M:%S)] gpu$g finished $n jobs" >> "$LOG"
}

for g in $(seq 0 $((NGPU-1))); do
  worker "$g" &
done
wait
echo "ALL DONE $(date +%H:%M:%S)"
