import argparse
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import pandas as pd
import os
from huggingface_hub import login
from embed_text_package.embed_text import Embedder
from dotenv import load_dotenv
import wandb

def get_embed(
    dataset,
    cols_to_be_embded = ['text'],
    bs = 1024,
    model_name="meta-llama/Meta-Llama-3-8B",
):
    embdr = Embedder()
    embdr.load(model_name)
    dataloader = DataLoader(dataset, batch_size=bs)
    emb = embdr.get_embeddings(
        dataloader, model_name, cols_to_be_embded
    )
    
    return emb['text']
    
def get_single_embed(search_path, z_path, hf_repo, bs=1024):
    search_df = pd.read_csv(search_path)
    z_df = pd.read_csv(z_path, usecols=["z"])
    deleted_col_indices = search_df[search_df.loc[:, "is_deleted"] == 1].index.tolist()
    
    text_df = search_df.loc[:, ["text"]]
    text_df = text_df.drop(deleted_col_indices).reset_index(drop=True)
    assert len(text_df) == len(z_df)
    
    tzpair_df = pd.concat([text_df, z_df], axis=1)
    tzpair_dataset = Dataset.from_pandas(tzpair_df)
    
    embed = get_embed(tzpair_dataset, bs=bs) # list of list
    assert len(embed) == len(text_df) == len(z_df)
    
    push_df = pd.DataFrame({
        'text': text_df['text'],
        'z': z_df['z'],
        'embed': embed
    })
    
    split_index = int(0.8 * len(push_df))
    push_train_df = push_df[:split_index]
    push_test_df = push_df[split_index:]
    
    push_train_dataset = Dataset.from_pandas(push_train_df.reset_index(drop=True))
    push_test_dataset = Dataset.from_pandas(push_test_df.reset_index(drop=True))
    
    push_dataset_dict = DatasetDict({
        'train': push_train_dataset,
        'test': push_test_dataset
    })
    push_dataset_dict.push_to_hub(hf_repo)

if __name__ == "__main__":
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--bs', type=int, default=1024)
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    get_single_embed(
        search_path = f'../data/pre_calibration/{args.dataset}/search.csv',
        z_path = f'../data/calibration/{args.dataset}/nonamor_z.csv',
        hf_repo = f'stair-lab/reeval_{args.dataset}-embed',
        bs=args.bs
    )

    