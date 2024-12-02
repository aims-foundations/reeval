import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


def delete_model_name(filename):
    filename = re.sub(r",stop=hash", "", filename)
    filename = re.sub(r",global_prefix=nlg", "", filename)
    return re.sub(r"model=[^,]*,?", "", filename).strip(",")


def extract_model_name(filename):
    match = re.search(r"model=([^,]*)", filename)
    return match.group(1)


def remove_duplicates(lst):
    seen = set()
    removed_indices = []
    for i, item in enumerate(lst):
        if item not in seen:
            seen.add(item)
        else:
            removed_indices.append(i)
    return removed_indices


def get_bool_answers(data):
    bool_answers = []
    for question in data["request_states"]:
        if "output_mapping" in question:
            # Step 1: Get the predicted answer from the model output, e.g., "B"
            predicted_answer = question["result"]["completions"][0]["text"].strip()
            # Step 2: Get the corresponding text for the predicted answer, Maps "B" to the actual text answer
            try:
                predicted_text = question["output_mapping"][predicted_answer]
            except KeyError:
                bool_answers.append(0)
                continue
            # Step 3: Loop through all choices
            for ref in question["instance"]["references"]:
                if ref["output"]["text"] == predicted_text:
                    matching_ref = ref
                    break
            # Step 4: If a matching reference is found, check if it is marked as correct
            if "correct" in matching_ref["tags"]:
                bool_answers.append(1)
            else:
                bool_answers.append(0)

        else:
            len_predicted_answers = len(question["result"]["completions"])
            predicted_answers = [
                question["result"]["completions"][i]["text"].strip()
                for i in range(len_predicted_answers)
            ]
            len_true_answers = len(question["instance"]["references"])
            true_answers = [
                question["instance"]["references"][i]["output"]["text"].strip()
                for i in range(len_true_answers)
            ]

            correct_tag = False
            for predicted_answer in predicted_answers:
                if predicted_answer in true_answers:
                    correct_tag = True
                    break
            bool_answers.append(int(correct_tag))

    return bool_answers


def get_bool_answers_logprob(data, threshold):
    bool_answers = []
    for question in data["request_states"]:
        # assert not question["instance"]["references"]
        logprob = question["result"]["completions"][0]["logprob"]
        bool_answers.append(int(logprob > threshold))
    return bool_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard",
        type=str,
        default="classic",
        choices=["classic", "mmlu", "thaiexam"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    # use wandb sweep, mmlu, thai_exam
    args = parser.parse_args()

    # data_folder = snapshot_download(
    #     repo_id=f"stair-lab/reeval_jsons", repo_type="dataset"
    # )
    data_folder = "../../data/gather_data/crawl_real"

    input_dir = f"{data_folder}/jsons/{args.dataset}_json"

    full_strings_all = pd.read_csv(
        f"{data_folder}/crawl_dataset_name_{args.leaderboard}.csv"
    )["Run"].tolist()
    full_strings = [
        f for f in full_strings_all if (f.split(":")[0].split(",")[0] == args.dataset)
    ]

    if args.dataset != "commonsense":
        full_strings = [f for f in full_strings if "ablation" not in f]
    if args.dataset == "truthful_qa":
        full_strings = [f for f in full_strings if "max_train_instances=0" not in f]

    all_model_names = list(set([extract_model_name(f) for f in full_strings]))
    all_model_names = sorted(all_model_names, key=lambda x: x[0])
    non_model_strings = list(set([delete_model_name(f) for f in full_strings]))
    print(non_model_strings)

    # Three types of questions:
    # 1. Multiple choice questions
    # 2. Open-ended questions
    # 3. Questions with no references with logprob

    logprob_tag = False
    with open(f"{input_dir}/{full_strings[0]}.json", "r") as f:
        data = json.load(f)

    if not data["request_states"][0]["instance"]["references"]:
        logprob_tag = True
        logprobs = []
        for full_string in full_strings:
            with open(f"{input_dir}/{full_string}.json", "r") as f:
                data = json.load(f)
            for question in data["request_states"]:
                logprob = question["result"]["completions"][0]["logprob"]
                logprobs.append(logprob)
        threshold = sum(logprobs) / len(logprobs)

    # response matrix
    max_lens = []
    max_len_file_names = []
    for i, non_model_string in enumerate(tqdm(non_model_strings)):
        max_len = 0
        max_len_file_name = ""
        single_matrix = {name: [] for name in all_model_names}

        for filename in tqdm(sorted(os.listdir(input_dir))):
            file_name_without_json = filename[:-5]
            if filename.endswith(".json") and (
                delete_model_name(file_name_without_json) == non_model_string
            ):
                model_name = extract_model_name(file_name_without_json)
                with open(f"{input_dir}/{filename}", "r") as f:
                    data = json.load(f)

                len_q = len(data["request_states"])
                if len_q > max_len:
                    max_len = len_q
                    max_len_file_name = filename

                if logprob_tag:
                    bool_answers = get_bool_answers_logprob(data, threshold)
                else:
                    bool_answers = get_bool_answers(data)
                single_matrix[model_name] = bool_answers

        for model_name, bool_answers in single_matrix.items():
            single_matrix[model_name] += [-1] * (
                max_len - len(single_matrix[model_name])
            )

        max_lens.append(max_len)
        max_len_file_names.append(max_len_file_name)

        single_matrix_df = pd.DataFrame(single_matrix).T
        single_matrix_df.columns = [f"{j}_{non_model_string}" for j in range(max_len)]

        assert single_matrix_df.index.tolist() == all_model_names

        if i == 0:
            all_matrix_df = single_matrix_df
        else:
            all_matrix_df = pd.concat([all_matrix_df, single_matrix_df], axis=1)

    # load all the text for each question
    search_dict = {"idx": [], "text": [], "is_deleted": []}
    base_idx = 0
    for i, non_model_string in enumerate(non_model_strings):
        with open(f"{input_dir}/{max_len_file_names[i]}", "r") as f:
            data = json.load(f)
        for j, question in enumerate(data["request_states"]):
            text = question["request"]["prompt"]
            search_dict["idx"].append(base_idx + j)
            search_dict["text"].append(text)
            search_dict["is_deleted"].append(0)
        base_idx += max_lens[i]

    # delete duplicate question text
    removed_indices = remove_duplicates(search_dict["text"])
    for idx in removed_indices:
        search_dict["is_deleted"][idx] = 1

    # delete questions that all models succeed/fail
    for idx, (col_name, col_data) in enumerate(all_matrix_df.items()):
        if set(col_data.unique()).issubset({0, -1}) or set(col_data.unique()).issubset(
            {1, -1}
        ):
            search_dict["is_deleted"][idx] = 1

    # delete "is_deleted" indices from all_matrix_df
    all_matrix_df = all_matrix_df.loc[
        :, all_matrix_df.columns[np.array(search_dict["is_deleted"]) == 0]
    ]

    # save data
    if args.dataset == "dyck_language_np=3":
        args.dataset = "dyck_language_np3"
    output_dir = f"../../data/pre_calibration/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    search_df = pd.DataFrame(search_dict)
    print(f"response matrix shape of {args.dataset}: {all_matrix_df.shape}")
    all_matrix_df.to_csv(f"{output_dir}/matrix.csv", index_label=None)

    search_df.to_csv(f"{output_dir}/search.csv", index=False, escapechar="\\")

    # Upload the content of the lbocal folder to your remote Space
    api = HfApi()
    api.upload_folder(
        folder_path=f"{output_dir}",
        path_in_repo=args.dataset,
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
    )
