#!/bin/bash

NUM_AGENTS=9
HOSTNAME=$(hostname)

for agent_num in $(seq 1 $((NUM_AGENTS))); do
    CUDA_VISIBLE_DEVICES=$agent_num nohup wandb agent ura-hcmut/plugin_regression_sweep_bydataset/4n7iaqtz > plugin_regression_sweep_bydataset_${HOSTNAME}_${agent_num}.log 2>&1 &
done
