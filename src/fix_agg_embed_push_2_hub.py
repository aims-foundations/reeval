from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
from utils import DATASETS
from sklearn.model_selection import train_test_split

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

    train_df, test_df = train_test_split(agg_df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    dataset_dict.push_to_hub("stair-lab/reeval_aggregate-embed")
