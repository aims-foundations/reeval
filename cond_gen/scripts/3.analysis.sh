#!/bin/bash

# List of dataset names
datasets=(
    "air-bench/air_bench_2024"
    "classic/babi_qa"
    "classic/bbq"
    # "classic/blimp"
    "classic/bold"
    "classic/boolq"
    "classic/civil_comments"
    "classic/code"
    "classic/commonsense"
    # "classic/copyright"
    # "classic/disinfo"
    "classic/dyck_language_np=3"
    "classic/entity_data_imputation"
    "classic/entity_matching"
    "classic/gsm"
    # "classic/ice"
    "classic/imdb"
    "classic/legal_support"
    "classic/lsat_qa"
    "classic/math"
    "classic/mmlu"
    "classic/msmarco"
    "classic/narrative_qa"
    # "classic/natural_qa"
    "classic/quac"
    "classic/raft"
    "classic/real_toxicity_prompts"
    # "classic/summarization_cnndm"
    "classic/summarization_xsum"
    "classic/synthetic_reasoning"
    "classic/synthetic_reasoning_natural"
    # "classic/the_pile"
    "classic/truthful_qa"
    "classic/twitter_aae"
    "classic/wikifact"
    "thaiexam/thai_exam"
)

# Generate questions
for dataset in "${datasets[@]}"; do
    echo "Runing SFT analysis on $dataset"
    python 3_sft_analysis.py --dataset $dataset --model stair-lab/reeval_Meta-Llama-3.1-8B-Instruct --run_generation 
done

# Choosing the most appropriate questions using amortized network for question difficulty
for dataset in "${datasets[@]}"; do
    echo "Runing SFT analysis on $dataset"
    python 3_sft_analysis.py --dataset $dataset --model stair-lab/reeval_Meta-Llama-3.1-8B-Instruct
done