import io
import os

import pandas as pd
import torch
from datasets import concatenate_datasets, Dataset, load_dataset
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm
from utils.constants import DATASETS

DATASETS = DATASETS[:1]

if __name__ == "__main__":
    upload_api = HfApi()

    # Combine response matrices
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    combined_matrix = None
    for dataset in tqdm(DATASETS):
        matrix = pd.read_csv(f"{data_folder}/{dataset}/matrix.csv", index_col=0)
        if combined_matrix is None:
            combined_matrix = matrix
        else:
            combined_matrix = combined_matrix.join(matrix, how="outer", rsuffix="_dup")
    combined_matrix.fillna(-1, inplace=True)
    # ds = Dataset.from_pandas(combined_matrix)
    # ds.push_to_hub("stair-lab/reeval_responses", "combined_data")
    combined_matrix_file = io.BytesIO()
    combined_matrix.to_csv(combined_matrix_file, index=False)
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

    # Combine column keys
    combined_search = None
    for dataset in DATASETS:
        matrix = pd.read_csv(f"{data_folder}/{dataset}/search.csv")
        if combined_search is None:
            combined_search = matrix
        else:
            combined_search = pd.concat([combined_search, matrix], axis=0)

    assert combined_matrix.shape[1] == (1 - combined_search["is_deleted"]).sum()
    # ds = Dataset.from_pandas(combined_search)
    # ds.push_to_hub("stair-lab/reeval_responses", "combined_data")
    combined_search_file = io.BytesIO()
    combined_search.to_csv(combined_search_file, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo="combined_data/search.csv",
        path_or_fileobj=combined_search_file,
    )

    combined_column_keys = None
    for dataset in DATASETS:
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

    # Combine the row keys
    combined_row_keys = None
    for dataset in DATASETS:
        matrix = pd.read_csv(f"{data_folder}/{dataset}/model_keys.csv")
        if combined_row_keys is None:
            combined_row_keys = matrix
        else:
            combined_row_keys = pd.concat([combined_row_keys, matrix], axis=0)
    # Remove the duplicates
    combined_row_keys = combined_row_keys.drop_duplicates(subset=["model_name"])

    assert combined_matrix.shape[0] == combined_row_keys.shape[0]
    df_sorted = (
        combined_row_keys.set_index("model_name")
        .loc[combined_matrix.index]
        .reset_index()
    )
    combined_row_keys = df_sorted.rename(columns={"index": "model_name"})
    
    # Set all values in `helm_score` column to nan
    combined_row_keys["helm_score"] = ""
    
    ctt_scores = []
    for ctts in combined_matrix.values:
        ctts = torch.tensor(ctts)
        ctt_scores.append(ctts[ctts!=-1].mean().item())
        
    combined_row_keys["ctt_score"] = ctt_scores
    
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

    # Combine the embedding
    emb_ds = None
    for dataset in DATASETS:
        data_files = os.listdir(f"{data_folder}/{dataset}")

        # Filter out the files that ends with .parquet
        data_files = [
            f for f in data_files if f.endswith(".parquet") and f.startswith("train")
        ]

        # Construct the full path
        data_files = [f"{data_folder}/{dataset}/{f}" for f in data_files]

        # Sort the files by the number
        data_files = sorted(data_files)

        dataset_emb = load_dataset("parquet", data_files=data_files, split="train")

        if emb_ds is None:
            emb_ds = dataset_emb
        else:
            emb_ds = concatenate_datasets([emb_ds, dataset_emb])

    assert len(emb_ds) == combined_matrix.shape[1]
    emb_ds.push_to_hub("stair-lab/reeval_responses", "combined_data")
