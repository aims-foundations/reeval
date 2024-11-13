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
    
    input_dir = f'../data/agg_embed/'
    
    agg_df = pd.concat(
        [pd.read_csv(f'{input_dir}/embed_{dataset}.csv') for dataset in DATASETS],
        ignore_index=True
    )
    agg_df['embed'] = agg_df['embed'].apply(lambda x: eval(x))
    
    agg_dataset = Dataset.from_pandas(agg_df)
    dataset_dict = DatasetDict({'train': agg_dataset})
    dataset_dict.push_to_hub("stair-lab/reeval_aggregate-embed")
