import io
import os

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from utils.constants import DATASETS


if __name__ == "__main__":
    upload_api = HfApi()

    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )

    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        data_files = os.listdir(f"{data_folder}/{dataset}")

        # Filter out the files that ends with .parquet
        data_files = [
            f for f in data_files if f.endswith(".parquet") and f.startswith("train")
        ]

        # Construct the full path
        data_files = [f"{data_folder}/{dataset}/{f}" for f in data_files]

        # Sort the files by the number
        data_files = sorted(data_files)

        print("Loading item embeddings...")
        dataset_emb = load_dataset("parquet", data_files=data_files, split="train")

        print("Converting to tensor...")
        item_embeddings = torch.tensor(dataset_emb["embed"], dtype=torch.float32)

        item_embedding_file = io.BytesIO()
        torch.save(item_embeddings, item_embedding_file)
        upload_api.upload_file(
            repo_id="stair-lab/reeval_responses",
            repo_type="dataset",
            path_in_repo=f"{dataset}/item_embeddings.pt",
            path_or_fileobj=item_embedding_file,
            # run_as_future=True,
        )
