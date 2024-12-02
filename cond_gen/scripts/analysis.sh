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
    echo "Runing SFT analysis on $dataset"
    python sft_analysis.py --dataset $dataset --run_generation
done