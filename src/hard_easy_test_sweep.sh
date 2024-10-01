#!/bin/bash

NUM_AGENTS=9
HOSTNAME=$(hostname)

for agent_num in $(seq 2 $((NUM_AGENTS))); do
    nohup wandb agent ura-hcmut/hard_easy_test_sweep/4la3ktxw > hard_easy_test_sweep_${HOSTNAME}_${agent_num}.log 2>&1 &
done
