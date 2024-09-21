#!/bin/bash

# wandb sweep save_json.yaml

NUM_AGENTS=3
HOSTNAME=$(hostname)

for agent_num in $(seq 1 $NUM_AGENTS); do
    nohup wandb agent yuhengtu/save_json/sqkkcqqp > save_json_${HOSTNAME}_${agent_num}.log 2>&1 &
done