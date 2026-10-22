#!/usr/bin/env bash
# finalize.sh -- when both queues are done, write the two analysis reports.
# Sits detached so the numbers exist even if nobody is watching. Idempotent:
# re-running just regenerates the .md files.
set -u
PY=$HOME/venvs/dl/bin/python
LD=/mnt/mydisk/xinguan-dendritic/dendritic-lab
BM=/mnt/mydisk/xinguan-dendritic/bench

while [ ! -f "$LD/DONE_ladgain" ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] ladder done, analysing" >> "$BM/finalize.log"
cd "$LD" || exit 1
$PY "$BM/analyze_ladgain.py" results_ladgain > "$BM/ladgain.md" 2>&1
echo "[$(date +%H:%M:%S)] wrote ladgain.md" >> "$BM/finalize.log"

while [ ! -f "$BM/DONE_d2wm" ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] d2wm done, analysing" >> "$BM/finalize.log"
cd "$BM" || exit 1
$PY "$BM/analyze_wm.py" results_d2_wm > "$BM/d2wm.md" 2>&1
echo "[$(date +%H:%M:%S)] wrote d2wm.md" >> "$BM/finalize.log"
touch "$BM/FINAL_DONE"
echo "[$(date +%H:%M:%S)] ALL FINALISED" >> "$BM/finalize.log"
