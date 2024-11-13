#!/bin/bash

NUM_AGENTS=8
HOSTNAME=$(hostname)

for agent_num in $(seq 2 $((NUM_AGENTS))); do
    CUDA_VISIBLE_DEVICES=$agent_num nohup wandb agent ura-hcmut/agg_amor_calibration_bydataset/6itzvduu > agg_amor_calibration_bydataset_${HOSTNAME}_${agent_num}.log 2>&1 &
done
