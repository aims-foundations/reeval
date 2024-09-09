#!/bin/bash

# wandb sweep sweep_config.yaml

NUM_AGENTS_PER_GPU=20
NUM_GPUS=10

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  for agent_num in $(seq 1 $NUM_AGENTS_PER_GPU); do
    CUDA_VISIBLE_DEVICES=$gpu nohup wandb agent yuhengtu/CAT_MLE/pjh2p4dw > ../data/synthetic/CAT_MLE/logs/agent_${gpu}_${agent_num}.log 2>&1 &
  done
done

wait
