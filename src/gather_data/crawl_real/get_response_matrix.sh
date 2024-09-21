#!/bin/bash

# wandb sweep get_response_matrix.yaml

NUM_AGENTS=3
HOSTNAME=$(hostname)

for agent_num in $(seq 1 $NUM_AGENTS); do
    nohup wandb agent yuhengtu/get_response_matrix/1co3lnz8 > get_response_matrix_${HOSTNAME}_${agent_num}.log 2>&1 &
done