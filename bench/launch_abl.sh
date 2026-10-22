#!/usr/bin/env bash
# launch_abl.sh -- the three controls that decide whether point_wmh's
# gap-independent 100.0 is real memory:
#   frozen gate  : keep every parameter, remove only the input dependence of g
#   open gate    : g = 1, i.e. no memory at all
#   no cue       : zero the cue channels at test time only
# Output goes to results_d2_abl so nothing overwrites the main curves.
cd /mnt/mydisk/xinguan-dendritic/bench || exit 1
PY=$HOME/venvs/dl/bin/python
: > abl.log
for lane in 0 1 2 3; do
  case $lane in
    0) S="0,1"; D=2 ;;
    1) S="2,3"; D=2 ;;
    2) S="4";   D=3 ;;
    3) S="5,6"; D=3 ;;
  esac
  CUDA_VISIBLE_DEVICES=$D $PY d2_xgap.py --train-gap 40 \
      --test-gaps 2,10,40,80,160,320 --seeds "$S" --taus 0.97 \
      --only-arms point_wm,point_wmh,dual_wmh --ablate \
      --outdir results_d2_abl >> abl.log 2>&1 &
done
wait
touch DONE_abl
