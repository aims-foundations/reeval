import argparse
import io

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from embed_text_package.embed_text_v2 import Embedder
from huggingface_hub import HfApi, snapshot_download
from torch.utils.data import DataLoader
from utils.constants import DATASETS, DESCRIPTION_MAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedder_name", type=str, default="meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    upload_api = HfApi()
    _, embedder_name = args.embedder_name.split("/")

    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    embdr = Embedder()
    embdr.load(args.embedder_name, tensor_parallel_size=num_gpus, dtype=torch.float16)
    for dataset in DATASETS[:-1]:  # skipping the combined dataset at the end
        print(f"Embedding {dataset}...")
        question_keys = pd.read_csv(f"{data_folder}/{dataset}/question_keys.csv")
        question_keys["text"] = (
            DESCRIPTION_MAP[dataset] + ", ### PROMPT: " + question_keys["text"]
        )
        text_dataset = Dataset.from_pandas(question_keys)

        dataloader = DataLoader(text_dataset, batch_size=args.batch_size)
        embed = embdr.get_embeddings(dataloader, args.embedder_name, ["text"])
        item_embeddings = torch.tensor(embed["text"], dtype=torch.float32)

        item_embedding_file = io.BytesIO()
        torch.save(item_embeddings, item_embedding_file)
        try:
            upload_api.delete_file(
                repo_id="stair-lab/reeval_responses",
                repo_type="dataset",
                path_in_repo=f"{dataset}/{embedder_name}_item_embeddings.pt",
            )
        except:
            pass
        try:
            upload_api.delete_file(
                repo_id="stair-lab/reeval_responses",
                repo_type="dataset",
                path_in_repo=f"{dataset}/item_embeddings.pt",
            )
        except:
            pass
        upload_api.upload_file(
            repo_id="stair-lab/reeval_responses",
            repo_type="dataset",
            path_in_repo=f"{dataset}/{embedder_name}_item_embeddings.pt",
            path_or_fileobj=item_embedding_file,
        )

    del embdr
    torch.cuda.empty_cache()

    # Combine the embedding
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )

    combined_item_embeddings = []
    for dataset in DATASETS[:-1]:
        emb_ds = torch.load(
            f"{data_folder}/{dataset}/{embedder_name}_item_embeddings.pt"
        )
        combined_item_embeddings.append(emb_ds)

    combined_item_embeddings = torch.cat(combined_item_embeddings, dim=0)

    item_embedding_file = io.BytesIO()
    torch.save(combined_item_embeddings, item_embedding_file)
    upload_api.delete_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo=f"combined_data/{embedder_name}_item_embeddings.pt",
    )
    try:
        upload_api.delete_file(
            repo_id="stair-lab/reeval_responses",
            repo_type="dataset",
            path_in_repo=f"combined_data/item_embeddings.pt",
        )
    except:
        pass

    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo=f"combined_data/{embedder_name}_item_embeddings.pt",
        path_or_fileobj=item_embedding_file,
    )
