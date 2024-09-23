import numpy as np
from tqdm import tqdm
from utils import DESCRIPTION_MAP, get_embed
import argparse
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv

def main(
    dataset,
    description,
    search_path,
    z_path, 
    save_path,
    bs=1024
):
    search_df = pd.read_csv(search_path)
    text_df = search_df.loc[search_df["is_deleted"] != 1, ["text"]].reset_index(drop=True)
    z_df = pd.read_csv(z_path, usecols=["z"])
    assert len(text_df) == len(z_df)
    
    text_df["text"] = description + ", ### PROMPT: " + text_df["text"]
    text_dataset = Dataset.from_pandas(text_df)
    embed = get_embed(text_dataset, bs=bs) # list of list
    assert len(embed) == len(text_df) == len(z_df)
    
    save_df = pd.DataFrame({
        'dataset': [dataset] * len(text_df),
        'text': text_df['text'],
        'z': z_df['z'],
        'embed': embed
    })
    save_df.to_csv(save_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024)
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    output_dir = f'../data/agg_embed/'
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset, desciption in tqdm(DESCRIPTION_MAP.items()):
        main(
            dataset=dataset,
            description=desciption,
            search_path=f'../data/pre_calibration/{dataset}/search.csv',
            z_path=f'../data/nonamor_calibration/{dataset}/nonamor_z.csv',
            save_path=f'{output_dir}/embed_{dataset}.csv',
            bs=args.bs
        )
    
    agg_df = pd.concat(
        [pd.read_csv(f'{output_dir}/embed_{dataset}.csv') for dataset in DESCRIPTION_MAP.keys()],
        ignore_index=True
    )
    agg_dataset = Dataset.from_pandas(agg_df)
    dataset_dict = DatasetDict({'train': agg_dataset})
    dataset_dict.push_to_hub("stair-lab/reeval_aggregate-embed")
