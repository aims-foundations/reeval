from tqdm import tqdm
from utils import DATASETS
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    input_dir = '/lfs/local/0/nqduc/.cache/huggingface/hub/datasets--stair-lab--reeval-agg_embed_folder/snapshots/07bcb8c88effd0fbd9b5811a0dc84235ebcdb1bf'
    
    dfs = []
    for dataset in tqdm(DATASETS):
        df = pd.read_csv(f'{input_dir}/embed_{dataset}.csv')
        embed_list = df['embed'].tolist()
        eval_embed_list = []
        for x in tqdm(embed_list):
            eval_embed_list.append(eval(x))
        df['embed'] = eval_embed_list
        dfs.append(df)

    agg_df = pd.concat(dfs, ignore_index=True)
    
    agg_dataset = Dataset.from_pandas(agg_df)
    dataset_dict = DatasetDict({'train': agg_dataset})
    dataset_dict.push_to_hub("stair-lab/reeval_aggregate-embed")
