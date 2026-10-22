#!/usr/bin/env bash
# launch_gain.sh -- after the lambda=3 probe: is the baseline's sigmoid GAIN the
# real confound? Tune g on a plain MLP+EWC at each task count's own best lambda.
cd "$(dirname "$0")" || exit 1
while [ ! -f DONE_l3 ]; do sleep 20; done
LOG=gain.log bash run_grid.sh jobs_gain.txt 2 > /dev/null 2>&1
touch DONE_gain
