import argparse
import pickle

import numpy as np
import pandas as pd
from datasets import Dataset
from embed_text_package.embed_text_v2 import Embedder
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from utils.constants import DESCRIPTION_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--fitting_method", type=str, default="mle")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    description = DESCRIPTION_MAP[args.dataset]

    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    search_df = pd.read_csv(f"{data_folder}/{args.dataset}/search.csv")

    text_df = search_df.loc[search_df["is_deleted"] != 1, ["text"]].reset_index(
        drop=True
    )

    item_parms_folder = snapshot_download(
        repo_id=f"stair-lab/reeval_{args.fitting_method}_calibration",
        repo_type="dataset",
    )
    item_parms = pickle.load(
        open(f"{item_parms_folder}/{args.PL}pl/{args.dataset}/item_parms.pkl", "rb")
    )
    # >>> n_questions x (3 + D)

    difficulty = np.array(item_parms)[:, 0].tolist()
    assert len(text_df) == len(difficulty)

    text_df["text"] = description + ", ### PROMPT: " + text_df["text"]
    text_dataset = Dataset.from_pandas(text_df)

    embdr = Embedder()
    embdr.load(args.model_name)
    dataloader = DataLoader(text_dataset, batch_size=args.batch_size)
    embed = embdr.get_embeddings(dataloader, args.model_name, ["text"])
    assert len(embed["text"]) == len(text_df) == len(difficulty)

    ds_embed = Dataset.from_dict(
        {
            "text": text_df["text"],
            "difficulty": difficulty,
            "embed": embed["text"],
        }
    )
    ds_embed.push_to_hub("stair-lab/reeval_all_embeddings", args.dataset)
