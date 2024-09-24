import argparse
import wandb
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm

if __name__ == "__main__":
    wandb.init(project="fix_agg_embed_push")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    input_dir = f'../data'
    
    df = pd.read_csv(f'{input_dir}/embed_{args.dataset}.csv')
    embed_list = df['embed'].tolist()
    eval_embed_list = []
    for x in tqdm(embed_list):
        eval_embed_list.append(eval(x))
    
    
    agg_dataset = Dataset.from_pandas(agg_df)
    dataset_dict = DatasetDict({'train': agg_dataset})
    dataset_dict.push_to_hub("stair-lab/reeval_aggregate-embed")

