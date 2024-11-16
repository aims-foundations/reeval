import argparse
import json
import os

import pandas as pd
import wandb
from datasets import Dataset
from utils import get_embed


def main(search_path, z_path, save_path, bs=1024):
    search_df = pd.read_csv(search_path)
    text_df = search_df.loc[search_df["is_deleted"] != 1, ["text"]].reset_index(
        drop=True
    )
    z_df = pd.read_csv(z_path, usecols=["z"])
    assert len(text_df) == len(z_df)

    text_dataset = Dataset.from_pandas(text_df)
    embed = get_embed(text_dataset, bs=bs)  # list of list
    assert len(embed) == len(text_df) == len(z_df)

    save_list = [
        {"text": text, "z": z, "embed": emb}
        for text, z, emb in zip(text_df["text"], z_df["z"], embed)
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_list, f, indent=4)


if __name__ == "__main__":
    wandb.init(project="get_embed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--bs", type=int, default=1024)
    args = parser.parse_args()

    output_dir = f"../data/get_embed/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    main(
        search_path=f"../data/pre_calibration/{args.dataset}/search.csv",
        z_path=f"../data/nonamor_calibration/{args.dataset}/nonamor_z.csv",
        save_path=f"{output_dir}/embed.json",
        bs=args.bs,
    )
