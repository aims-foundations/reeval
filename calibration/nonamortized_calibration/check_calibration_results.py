
import os
DATASETS = [
    "airbench",
    "twitter_aae",
    "math",
    "entity_data_imputation",
    "real_toxicity_prompts",
    "civil_comments",
    "imdb",
    "boolq",
    "wikifact",
    "babi_qa",
    "mmlu",
    "truthful_qa",
    "legal_support",
    "synthetic_reasoning",
    "quac",
    "entity_matching",
    "synthetic_reasoning_natural",
    "bbq",
    "raft",
    "narrative_qa",
    "commonsense",
    "lsat_qa",
    "bold",
    "dyck_language_np3",
    "combined_data",
]

D = [1, 2]
KL = [1, 2, 3]
FITTING_METHODS = ["mle", "em"]
AMORTIZED_QUESTION = [True, False]
AMORTIZED_STUDENT = [True, False]

for dataset in DATASETS:
    for d in D:
        for kl in KL:
            for fitting_method in FITTING_METHODS:
                for amortized_question in AMORTIZED_QUESTION:
                    for amortized_student in AMORTIZED_STUDENT:
                        output_dir = f"../../results/calibration/{dataset}/s42_{fitting_method}_{kl}pl_{d}d{'_aq' if amortized_question else ''}{'_as' if amortized_student else ''}"

                        if not os.path.exists(f"{output_dir}/test_indices.pkl") or \
                            not os.path.exists(f"{output_dir}/abilities.pkl") or \
                            not os.path.exists(f"{output_dir}/item_parms.pkl") or \
                            not os.path.exists(f"{output_dir}/train_indices.pkl"):
                            print("Missing files in", output_dir)
