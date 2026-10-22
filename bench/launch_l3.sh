#!/usr/bin/env bash
# launch_l3.sh -- after the sigmoid probe, probe lambda=3 at T=20.
cd "$(dirname "$0")" || exit 1
while [ ! -f DONE_sig ]; do sleep 20; done
bash launch_pair.sh jobs_l3_a.txt jobs_l3_b.txt DONE_l3
