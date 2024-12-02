#!/bin/bash

# List of dataset names
datasets=(
    "airbench"
    "twitter_aae"
    "math"
    "entity_data_imputation"
    "real_toxicity_prompts"
    "civil_comments"
    "imdb"
    "boolq"
    "wikifact"
    "babi_qa"
    "mmlu"
    "truthful_qa"
    "legal_support"
    "synthetic_reasoning"
    "quac"
    "entity_matching"
    "synthetic_reasoning_natural"
    "bbq"
    "raft"
    "narrative_qa"
    "commonsense"
    "lsat_qa"
    "bold"
    "dyck_language_np3"
    "thai_exam"
)

for dataset in "${datasets[@]}"; do
    echo "Running $dataset"
    python 1_sft_dataset.py --dataset $dataset --model meta-llama/Meta-Llama-3-8B
done

for dataset in "${datasets[@]}"; do
    echo "Running $dataset"
    python 1_sft_dataset.py --dataset $dataset --model mistralai/Mistral-7B-Instruct-v0.3
done

for dataset in "${datasets[@]}"; do
    echo "Creating PPO dataset $dataset"
    python ppo_dataset.py --dataset $dataset
done