#!/bin/bash

NUM_AGENTS=25
HOSTNAME=$(hostname)

for agent_num in $(seq 1 $NUM_AGENTS); do
    nohup wandb agent yuhengtu/cat/xmpv382d > cat_${HOSTNAME}_${agent_num}.log 2>&1 &
done
