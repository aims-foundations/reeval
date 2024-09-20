#!/bin/bash

# wandb sweep get_response_matrix.yaml

NUM_AGENTS=8
HOSTNAME=$(hostname)

for agent_num in $(seq 1 $NUM_AGENTS); do
    nohup wandb agent ura-hcmut/get_response_matrix/8njj5am5 > get_response_matrix_${HOSTNAME}_${agent_num}.log 2>&1 &
done