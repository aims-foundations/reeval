#!/bin/bash

NUM_AGENTS=8
HOSTNAME=$(hostname)

for agent_num in $(seq 2 $((NUM_AGENTS))); do
    CUDA_VISIBLE_DEVICES=$agent_num nohup wandb agent ura-hcmut/agg_amor_calibration_byrandom/k0u4i42q > agg_amor_calibration_byrandom_${HOSTNAME}_${agent_num}.log 2>&1 &
done
