#!/usr/bin/env bash
# launch_low.sh -- wait for the T=20 curve, then fill in the low-lambda tail
# (lambda = 10, 30). The T=5/T=10 curves bottomed out at the old grid floor,
# so the grid has to be extended downwards before lambda* can be trusted.
cd "$(dirname "$0")" || exit 1
while [ ! -f DONE_t20 ]; do sleep 30; done
for nt in 5 10 20; do
  bash launch_pair.sh "jobs_low${nt}_a.txt" "jobs_low${nt}_b.txt" "DONE_low${nt}"
done
touch DONE_LOW
