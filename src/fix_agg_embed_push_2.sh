#!/bin/bash

NUM_AGENTS=24
HOSTNAME=$(hostname)

for agent_num in $(seq 1 $NUM_AGENTS); do
    nohup wandb agent yuhengtu/fix_agg_embed_push_2/r5ekbewk > fix_agg_embed_push_2_${HOSTNAME}_${agent_num}.log 2>&1 &
done