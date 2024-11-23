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


def check_results(output_dir, amortized_question, amortized_student):
    if (
        not os.path.exists(f"{output_dir}/train_question_indices.pkl")
        or not os.path.exists(f"{output_dir}/test_question_indices.pkl")
        or not os.path.exists(f"{output_dir}/train_model_indices.pkl")
        or not os.path.exists(f"{output_dir}/test_model_indices.pkl")
        or not os.path.exists(f"{output_dir}/abilities.pkl")
        or not os.path.exists(f"{output_dir}/item_parms.pkl")
    ):
        print("Missing files in", output_dir)
        return False
    elif amortized_question:
        if not os.path.exists(
            f"{output_dir}/item_parameters_nn.pkl"
        ) or not os.path.exists(f"{output_dir}/item_parameters_nn.pt"):
            print("Missing files in", output_dir)
            return False
    elif amortized_student:
        if not os.path.exists(
            f"{output_dir}/student_parameters_nn.pkl"
        ) or not os.path.exists(f"{output_dir}/student_parameters_nn.pt"):
            print("Missing files in", output_dir)
            return False
    return True


if __name__ == "__main__":
    for dataset in DATASETS:
        for d in D:
            for kl in KL:
                for fitting_method in FITTING_METHODS:
                    for amortized_question in AMORTIZED_QUESTION:
                        for amortized_student in AMORTIZED_STUDENT:
                            output_dir = f"../results/calibration/{dataset}/s42_{fitting_method}_{kl}pl_{d}d{'_aq' if amortized_question else ''}{'_as' if amortized_student else ''}"
                            check_results(
                                output_dir, amortized_question, amortized_student
                            )
