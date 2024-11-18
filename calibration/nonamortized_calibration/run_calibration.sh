#!/bin/bash


wandb sweep calibration.yaml
CUDA_VISIBLE_DEVICES=5 wandb agent ura-hcmut/calibration/ixc900c7 &
CUDA_VISIBLE_DEVICES=4 wandb agent ura-hcmut/calibration/ixc900c7 &
CUDA_VISIBLE_DEVICES=6 wandb agent ura-hcmut/calibration/ixc900c7 &
CUDA_VISIBLE_DEVICES=7 wandb agent ura-hcmut/calibration/ixc900c7 &
wandb agent ura-hcmut/calibration/ixc900c7
wandb agent ura-hcmut/calibration/ixc900c7

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
    "combined_data"
)

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Lopp through each PL
    for PL in 1 2 3; do
        # Loop through each dimension d
        for d in 1 2; do
            # Loop through each fitting method
            for fitting_method in mle em;
                # Loop through whether to use amortization on question
                for amortize_question in True False; do
                    echo "Calibrating $dataset $PL $d $fitting_method $amortize_question"
                    python calibrate.py --dataset $dataset --PL $PL --d $d --fitting_method $fitting_method --amortize_question $amortize_question &
                done
            done
        done
    done
done

# python calibration_analysis.py --PL 1 --fitting_method mle
# python calibration_analysis.py --PL 1 --fitting_method mle
# python calibration_analysis.py --PL 1 --fitting_method em
# python calibration_analysis.py --PL 1 --fitting_method em