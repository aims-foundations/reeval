#!/bin/bash

NUM_AGENTS=9
HOSTNAME=$(hostname)

for agent_num in $(seq 0 $((NUM_AGENTS-1))); do
    CUDA_VISIBLE_DEVICES=$agent_num nohup wandb agent ura-hcmut/plugin_regression_sweep/92qmfag7 > plugin_regression_sweep_${HOSTNAME}_${agent_num}.log 2>&1 &
done
