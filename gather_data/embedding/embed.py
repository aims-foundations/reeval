import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from embed_text_package.embed_text_v2 import Embedder
from huggingface_hub import HfApi, snapshot_download
from torch.utils.data import DataLoader
from utils.constants import DESCRIPTION_MAP
import io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--fitting_method", type=str, default="mle")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    upload_api = HfApi()
    
    num_gpus = torch.cuda.device_count()

    description = DESCRIPTION_MAP[args.dataset]
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    search_df = pd.read_csv(f"{data_folder}/{args.dataset}/search.csv")

    text_df = search_df.loc[search_df["is_deleted"] != 1, ["text"]].reset_index(
        drop=True
    )

    # item_parms_folder = snapshot_download(
    #     repo_id=f"stair-lab/reeval_{args.fitting_method}_calibration",
    #     repo_type="dataset",
    # )
    # item_parms = pickle.load(
    #     open(f"{item_parms_folder}/{args.PL}pl/{args.dataset}/item_parms.pkl", "rb")
    # )
    # # >>> n_questions x (3 + D)

    # difficulty = np.array(item_parms)[:, 0].tolist()
    # assert len(text_df) == len(difficulty)

    text_df["text"] = description + ", ### PROMPT: " + text_df["text"]
    text_dataset = Dataset.from_pandas(text_df)

    embdr = Embedder()
    embdr.load(args.model_name, tensor_parallel_size=num_gpus, dtype=torch.float16)
    dataloader = DataLoader(text_dataset, batch_size=args.batch_size)
    embed = embdr.get_embeddings(dataloader, args.model_name, ["text"])
    # assert len(embed["text"]) == len(text_df) == len(difficulty)

    item_embeddings = torch.tensor(embed["text"], dtype=torch.float32)

    item_embedding_file = io.BytesIO()
    torch.save(item_embeddings, item_embedding_file)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo=f"{args.dataset}/item_embeddings.pt",
        path_or_fileobj=item_embedding_file,
        # run_as_future=True,
    )

    text_file = io.BytesIO()
    row_key.to_csv(model_key_file, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_responses",
        repo_type="dataset",
        path_in_repo=f"{dataset}/model_keys.csv",
        path_or_fileobj=model_key_file,
        # run_as_future=True,
    )


    ds_embed = Dataset.from_dict(
        {
            "text": text_df["text"],
            # "difficulty": difficulty,
            # "embed": embed["text"],
        }
    )
    ds_embed.push_to_hub("stair-lab/reeval_all_embeddings", args.dataset)
