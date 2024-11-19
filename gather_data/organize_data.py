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
    model_ids = pd.read_csv(f"{model_info_folder}/model_id.csv", index_col=0)
    model_info = pd.read_csv(f"{model_info_folder}/model_id_final.csv", index_col=0)

    huggingface_model_ids = {}
    for _, model_name in model_ids["model_names"].items():
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

    embedding_folder = snapshot_download(
        repo_id="stair-lab/reeval_all_embeddings", repo_type="dataset"
    )

    helm_score_folder = snapshot_download(
        repo_id="stair-lab/reeval_helm_scores", repo_type="dataset"
    )

    for dataset in tqdm(DATASETS):
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
            # run_as_future=True,
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

        for file in os.listdir(f"{embedding_folder}/{dataset}"):
            embedding_file = f"{embedding_folder}/{dataset}/{file}"
            upload_api.upload_file(
                repo_id="stair-lab/reeval_responses",
                repo_type="dataset",
                path_in_repo=f"{dataset}/{file}",
                path_or_fileobj=embedding_file,
                # run_as_future=True,
            )

    print("Done")
