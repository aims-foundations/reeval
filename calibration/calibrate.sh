#!/bin/bash

# List of datasets
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
    "combined_data"
)

# Iterate through each dataset and run the calibrate.py script
for dataset in "${datasets[@]}"
do
    echo "Running calibration for dataset: $dataset"
    python calibrate.py --dataset "$dataset" --D 1 --PL 1 --max_epoch 1000 &
done
