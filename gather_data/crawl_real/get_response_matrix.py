import argparse
import json
import os
import re

import pandas as pd
from dataset_info_stats import delete_model_name
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


def extract_model_name(filename):
    match = re.search(r"model=([^,]*)", filename)
    return match.group(1)


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
        assert not question["instance"]["references"]
        logprob = question["result"]["completions"][0]["logprob"]
        bool_answers.append(int(logprob > threshold))
    return bool_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard", type=str, default="classic", choices=["classic", "mmlu", "thaiexam"]
    )
    parser.add_argument("--dataset", type=str, required=True)  # use wandb sweep, mmlu, thai_exam
    args = parser.parse_args()

    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_jsons", repo_type="dataset"
    )
    # data_folder = "../../data/gather_data/crawl_real"

    input_dir = f"{data_folder}/jsons/{args.dataset}_json"
    output_dir = f"../../data/pre_calibration/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    full_strings_all = pd.read_csv(
        f"{data_folder}/crawl_dataset_name_{args.leaderboard}.csv"
    )["Run"].tolist()
    full_strings = [
        f for f in full_strings_all if (f.split(":")[0].split(",")[0] == args.dataset)
    ]
    all_model_names = list(set([extract_model_name(f) for f in full_strings]))
    all_model_names = sorted(all_model_names, key=lambda x: x[0])
    non_model_strings = list(set([delete_model_name(f) for f in full_strings]))

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

        for filename in tqdm(os.listdir(input_dir)):
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

        if i == 0:
            all_matrix_df = single_matrix_df
        else:
            all_matrix_df = pd.concat([all_matrix_df, single_matrix_df], axis=1)

    assert all_matrix_df.shape[0] == len(all_model_names)

    bool_delete_list = []
    for col_name, col_data in all_matrix_df.items():
        if set(col_data.unique()).issubset({0, -1}) or set(col_data.unique()).issubset(
            {1, -1}
        ):
            all_matrix_df = all_matrix_df.drop(columns=[col_name])
            bool_delete_list.append(1)
        else:
            bool_delete_list.append(0)

    all_matrix_df.to_csv(f"{output_dir}/matrix.csv", index_label=None)

    # index search
    search_list = []
    base_idx = 0
    for i, non_model_string in enumerate(non_model_strings):
        with open(f"{input_dir}/{max_len_file_names[i]}", "r") as f:
            data = json.load(f)
        for j, question in enumerate(data["request_states"]):
            text = question["instance"]["input"]["text"]
            search_list.append([base_idx + j, text, bool_delete_list[base_idx + j]])
        base_idx += max_lens[i]

    search_df = pd.DataFrame(search_list, columns=["idx", "text", "is_deleted"])
    search_df.to_csv(f"{output_dir}/search.csv", index=False, escapechar="\\")

    # Upload the content of the local folder to your remote Space
    api = HfApi()
    api.upload_folder(
        folder_path=f"{output_dir}",
        path_in_repo=args.dataset,
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
    )
