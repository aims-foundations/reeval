from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from utils import DATASETS

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)
    
    input_dir = '/lfs/local/0/nqduc/.cache/huggingface/hub/datasets--stair-lab--reeval-agg_embed_folder/snapshots/07bcb8c88effd0fbd9b5811a0dc84235ebcdb1bf'
    output_dir = f'{input_dir}/new'

    agg_df = pd.concat(
        [pd.read_csv(f'{output_dir}/new_embed_{dataset}.csv') for dataset in DATASETS],
        ignore_index=True
    )
    
    agg_df['embed'] = agg_df[[f'embed_{i}' for i in range(4096)]].values.tolist()
    agg_df = agg_df.drop(columns=[f'embed_{i}' for i in range(4096)])

    grouped_datasets = {}
    for dataset_name, group in agg_df.groupby('dataset'):
        hf_dataset = Dataset.from_pandas(group.drop(columns=['dataset']))
        grouped_datasets[dataset_name] = hf_dataset

    dataset_dict = DatasetDict(grouped_datasets)
    dataset_dict.push_to_hub("stair-lab/reeval_aggregate-embed")
