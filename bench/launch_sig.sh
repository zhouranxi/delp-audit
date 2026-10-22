#!/usr/bin/env bash
# launch_sig.sh -- after the ladder queue drains, run the sigmoid-gain probe.
cd "$(dirname "$0")" || exit 1
while [ ! -f DONE_LOW ]; do sleep 30; done
mkdir -p results_sig
bash launch_pair.sh jobs_sig_a.txt jobs_sig_b.txt DONE_sig
