import io
import os

import pandas as pd
import requests
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, snapshot_download

from tqdm import tqdm
from utils.constants import DATASETS


def check_model_existence(huggingface_model_id):
    # call model in huggingface in a try catch block
    model_url = f"https://huggingface.co/{huggingface_model_id}"

    # Check if model exists by calling the model url
    # If returning code is 200, then model exists
    try:
        response = requests.get(model_url)
        if response.status_code != 200:
            print(f"Model {huggingface_model_id} does not exist")
            huggingface_model_id = None
    except:
        print(f"Model {huggingface_model_id} does not exist")
        huggingface_model_id = None

    return huggingface_model_id


if __name__ == "__main__":
    upload_api = HfApi()

    # Combine response matrices
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )

    model_info_folder = snapshot_download(
        repo_id="stair-lab/reeval_model_info", repo_type="dataset"
    )
    # model_ids = pd.read_csv(f"{model_info_folder}/model_id.csv", index_col=0)
    model_ids = []
    for dataset in DATASETS[:-1]:
        model_name = pd.read_csv(
            f"{data_folder}/{dataset}/matrix.csv", index_col=0
        ).index.tolist()
        model_ids.extend(model_name)
    model_ids = sorted(list(set(model_ids)))
    model_info = pd.read_csv(f"{model_info_folder}/model_id_final.csv", index_col=0)

    huggingface_model_ids = {}
    # for _, model_name in model_ids["model_names"].items():
    for model_name in model_ids:
        hf_model_name = model_name.replace("_", "/")
        if hf_model_name == "mistralai/mixtral-8x22b":
            hf_model_name = "mistralai/Mixtral-8x22B-v0.1"
        elif hf_model_name == "NousResearch/Nous-Capybara-7B-V1p9":
            hf_model_name = "NousResearch/Nous-Capybara-7B-V1.9"
        elif hf_model_name == "cohere/command-r":
            hf_model_name = "CohereForAI/c4ai-command-r-v01"
        elif hf_model_name == "cohere/command-r-plus":
            hf_model_name = "CohereForAI/c4ai-command-r-plus"
        elif hf_model_name == "mistralai/mixtral-8x7b-32kseqlen":
            hf_model_name = "mistralai/Mixtral-8x7B-v0.1"
        elif hf_model_name == "teknium/OpenHermes-2p5-Mistral-7B":
            hf_model_name = "teknium/OpenHermes-2.5-Mistral-7B"
        elif hf_model_name == "meta/llama-3-70b":
            hf_model_name = "meta-llama/Meta-Llama-3-70B"
        elif hf_model_name == "meta/llama-3-8b":
            hf_model_name = "meta-llama/Meta-Llama-3-8B"
        elif hf_model_name == "meta/llama-3.1-405b-instruct-turbo":
            hf_model_name = "meta-llama/Llama-3.1-405B-Instruct"
        elif hf_model_name == "meta/llama-3.1-70b-instruct-turbo":
            hf_model_name = "meta-llama/Llama-3.1-70B-Instruct"
        elif hf_model_name == "meta/llama-3.1-8b-instruct-turbo":
            hf_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif hf_model_name == "ai21/jamba-instruct":
            hf_model_name = "ai21labs/Jamba-v0.1"
        elif hf_model_name == "ai21/jamba-1.5-mini":
            hf_model_name = "ai21labs/AI21-Jamba-1.5-Mini"
        elif hf_model_name == "ai21/jamba-1.5-large":
            hf_model_name = "ai21labs/AI21-Jamba-1.5-Large"
        elif hf_model_name == "together/gpt-j-6b":
            hf_model_name = "togethercomputer/GPT-JT-6B-v1"
        elif hf_model_name == "together/gpt-neox-20b":
            hf_model_name = "togethercomputer/GPT-NeoXT-Chat-Base-20B"
        elif hf_model_name == "together/redpajama-incite-base-3b-v1":
            hf_model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
        elif hf_model_name == "together/redpajama-incite-base-7b":
            hf_model_name = "togethercomputer/RedPajama-INCITE-7B-Base"
        elif hf_model_name == "together/redpajama-incite-instruct-3b-v1":
            hf_model_name = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
        elif hf_model_name == "together/redpajama-incite-instruct-7b":
            hf_model_name = "togethercomputer/RedPajama-INCITE-7B-Instruct"

        try:
            hf_org_name, hf_model_name = hf_model_name.split("/")
        except:
            print(f"Splitting the name of model {hf_model_name} failed")
            hf_org_name = None
            huggingface_model_id = None

        if hf_org_name == "meta":
            hf_org_name = "meta-llama"

        huggingface_model_id = check_model_existence(f"{hf_org_name}/{hf_model_name}")
        huggingface_model_ids[model_name] = huggingface_model_id

    helm_score_folder = snapshot_download(
        repo_id="stair-lab/reeval_helm_scores", repo_type="dataset"
    )

    for dataset in tqdm(DATASETS[:-1]):  # skipping the combined dataset at the end
        # Data
        data = pd.read_csv(f"{data_folder}/{dataset}/matrix.csv", index_col=0)
        response_matrix = torch.tensor(data.values, dtype=torch.float32)

        # Push response matrix as a torch object
        response_matrix_file = io.BytesIO()
        torch.save(response_matrix, response_matrix_file)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_responses",
            repo_type="dataset",
            path_in_repo=f"{dataset}/response_matrix.pt",
            path_or_fileobj=response_matrix_file,
        )

        if os.path.exists(f"{helm_score_folder}/{dataset}.csv"):
            helm_scores = pd.read_csv(f"{helm_score_folder}/{dataset}.csv")
        else:
            helm_scores = None

        # Row key
        row_key = data.index.tolist()
        row_data = []

        assert len(row_key) == response_matrix.shape[0]
        for model_name, model_ctt_scores in zip(row_key, response_matrix):
            single_model_info = model_info[
                model_info["model_names_reeval"] == model_name
            ]
            if helm_scores is not None:
                single_model_hs = helm_scores[helm_scores["model_name"] == model_name]

                if len(single_model_hs) == 0:
                    single_model_hs = None
                else:
                    single_model_hs = single_model_hs["score"].iloc[0]
            else:
                single_model_hs = None

            model_ctt_score = model_ctt_scores[model_ctt_scores != -1].mean().item()

            if len(single_model_info) == 0:
                row_data.append(
                    {
                        "model_name": model_name,
                        "huggingface_model_id": huggingface_model_ids[model_name],
                        "model_size": None,
                        "pretraining_data_size": None,
                        "flop": None,
                        "helm_score": single_model_hs,
                        "ctt_score": model_ctt_score,
                    }
                )
            else:
                row_data.append(
                    {
                        "model_name": model_name,
                        "huggingface_model_id": huggingface_model_ids[model_name],
                        "model_size": single_model_info["Model Size (B)"].values[0],
                        "pretraining_data_size": single_model_info[
                            "Pretraining Data Size (T)"
                        ].values[0],
                        "flop": single_model_info["FLOPs (1E21)"].values[0],
                        "helm_score": single_model_hs,
                        "ctt_score": model_ctt_score,
                    }
                )

        row_key = pd.DataFrame(row_data)

        assert row_key.shape[0] == data.shape[0]
        # row_key.to_csv(f"{data_folder}/{dataset}/model_keys.csv", index=False)
        model_key_file = io.BytesIO()
        row_key.to_csv(model_key_file, index=False)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_responses",
            repo_type="dataset",
            path_in_repo=f"{dataset}/model_keys.csv",
            path_or_fileobj=model_key_file,
            # run_as_future=True,
        )

        # Column key
        search_df = pd.read_csv(f"{data_folder}/{dataset}/search.csv", index_col=0)
        question_content_df = search_df.loc[
            search_df["is_deleted"] != 1, ["text"]
        ].reset_index(drop=True)
        column_key = pd.DataFrame(
            {"question_id": data.columns.tolist(), "text": question_content_df["text"]}
        )
        assert column_key.shape[0] == data.shape[1]
        # column_key.to_csv(f"{data_folder}/{dataset}/question_keys.csv", index=False)
        question_key_file = io.BytesIO()
        column_key.to_csv(question_key_file, index=False)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_responses",
            repo_type="dataset",
            path_in_repo=f"{dataset}/question_keys.csv",
            path_or_fileobj=question_key_file,
            # run_as_future=True,
        )

    # Combine response matrices
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    combined_matrix = None
    for dataset in tqdm(DATASETS[:-1]):  # skipping the combined dataset at the end
        matrix = pd.read_csv(f"{data_folder}/{dataset}/matrix.csv", index_col=0)
        if combined_matrix is None:
            combined_matrix = matrix
        else:
            combined_matrix = combined_matrix.join(matrix, how="outer", rsuffix="_dup")
    combined_matrix.fillna(-1, inplace=True)

    print(f"Combined matrix shape: {combined_matrix.shape}")

    # Combine the row keys
    combined_row_keys = None
    for dataset in DATASETS[:-1]:
        matrix = pd.read_csv(f"{data_folder}/{dataset}/model_keys.csv")
        if combined_row_keys is None:
            combined_row_keys = matrix
        else:
            combined_row_keys = pd.concat([combined_row_keys, matrix], axis=0)

    # Remove the duplicates
    combined_row_keys = combined_row_keys.drop_duplicates(subset=["model_name"])

    df_sorted = (
        combined_row_keys.set_index("model_name")
        .loc[combined_matrix.index]
        .reset_index()
    )
    combined_row_keys = df_sorted.rename(columns={"index": "model_name"})

    assert (
        combined_matrix.index.tolist() == combined_row_keys["model_name"].tolist()
    ), f"{combined_matrix.index.tolist()} != {combined_row_keys['model_name'].tolist()}"

    # upload the response matrix as a csv file
    combined_matrix_file = io.BytesIO()
    combined_matrix.to_csv(combined_matrix_file, index_label=None)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo="combined_data/matrix.csv",
        path_or_fileobj=combined_matrix_file,
    )

    # upload the response matrix as a torch object
    combined_matrix_file = io.BytesIO()
    torch.save(
        torch.tensor(combined_matrix.values, dtype=torch.float32), combined_matrix_file
    )
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo="combined_data/response_matrix.pt",
        path_or_fileobj=combined_matrix_file,
    )

    # ds = Dataset.from_pandas(combined_row_keys)
    # ds.push_to_hub("stair-lab/reeval_responses", "combined_data")
    combined_row_file = io.BytesIO()
    combined_row_keys.to_csv(combined_row_file, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo="combined_data/model_keys.csv",
        path_or_fileobj=combined_row_file,
    )

    # Combine column keys
    combined_search = None
    for dataset in DATASETS[:-1]:
        matrix = pd.read_csv(f"{data_folder}/{dataset}/search.csv")
        if combined_search is None:
            combined_search = matrix
        else:
            combined_search = pd.concat([combined_search, matrix], axis=0)

    assert combined_matrix.shape[1] == (1 - combined_search["is_deleted"]).sum()

    # upload the combined search matrix as a csv file
    combined_search_file = io.BytesIO()
    combined_search.to_csv(combined_search_file, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo="combined_data/search.csv",
        path_or_fileobj=combined_search_file,
    )

    combined_column_keys = None
    for dataset in DATASETS[:-1]:
        matrix = pd.read_csv(f"{data_folder}/{dataset}/question_keys.csv")
        if combined_column_keys is None:
            combined_column_keys = matrix
        else:
            combined_column_keys = pd.concat([combined_column_keys, matrix], axis=0)

    assert combined_matrix.shape[1] == combined_column_keys.shape[0]
    df_sorted = (
        combined_column_keys.set_index("question_id")
        .loc[combined_matrix.columns]
        .reset_index()
    )
    combined_column_keys = df_sorted.rename(columns={"index": "question_id"})
    # ds = Dataset.from_pandas(combined_column_keys)
    # ds.push_to_hub("stair-lab/reeval_responses", "combined_data")
    combined_column_file = io.BytesIO()
    combined_column_keys.to_csv(combined_column_file, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo="combined_data/question_keys.csv",
        path_or_fileobj=combined_column_file,
    )
