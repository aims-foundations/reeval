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
)

# Loop through each dataset and run the Python command
for dataset in "${datasets[@]}"; do
    echo "Calibrating $dataset 1PL"
    python calibrate.py --dataset $dataset --PL 1 & 
done

for dataset in "${datasets[@]}"; do
    echo "Calibrating $dataset 2PL"
    python calibrate.py --dataset $dataset --PL 2 & 
done

python calibrate_analysis.py --PL 1
python calibrate_analysis.py --PL 1