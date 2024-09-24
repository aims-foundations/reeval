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
    
    input_dir = '/lfs/local/0/nqduc/.cache/huggingface/hub/datasets--stair-lab--reeval-agg_embed_folder/snapshots/07bcb8c88effd0fbd9b5811a0dc84235ebcdb1bf'
    output_dir = f'{input_dir}/new'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(f'{input_dir}/embed_{args.dataset}.csv')
    embed_list = df['embed'].tolist()
    
    eval_embed_list = []
    for x in tqdm(embed_list):
        eval_embed_list.append(eval(x))

    embed_df = pd.DataFrame(eval_embed_list, columns=[f'embed_{i}' for i in range(4096)])
    result_df = pd.concat([df[['dataset', 'text', 'z']], embed_df], axis=1)
    result_df.to_csv(f'{output_dir}/new_embed_{args.dataset}.csv', index=False)
