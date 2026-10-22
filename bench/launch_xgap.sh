#!/usr/bin/env bash
# launch_xgap.sh -- cross-gap generalisation test for D2.
# Train at a single gap, score the same weights at every test gap. Lives on
# gpu2/gpu3 because the ladder grid owns gpu0/gpu1 and these jobs need <1 GB.
cd /mnt/mydisk/xinguan-dendritic/bench || exit 1
PY=$HOME/venvs/dl/bin/python
: > xgap.log
for lane in 0 1 2 3; do
  case $lane in
    0) S="0,1";     D=2 ;;
    1) S="2,3";     D=2 ;;
    2) S="4";       D=3 ;;
    3) S="5,6";     D=3 ;;
  esac
  CUDA_VISIBLE_DEVICES=$D $PY d2_xgap.py --train-gap 40 \
      --test-gaps 2,10,40,80,160,320 --seeds "$S" --taus 0.97 \
      --outdir results_d2_xgap >> xgap.log 2>&1 &
done
wait
touch DONE_xgap
