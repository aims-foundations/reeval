import argparse
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import pandas as pd
import os
import json
from huggingface_hub import login
from dotenv import load_dotenv
from utils import DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['individual', 'aggregate'])
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)

    dataset_dict = {}
    for dataset in tqdm(DATASETS):
        with open(f'../data/embed_{args.task}/{dataset}/embed.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            dataset_split = Dataset.from_pandas(df)
            dataset_dict[dataset] = dataset_split

    hf_dataset_dict = DatasetDict(dataset_dict)
    hf_dataset_dict.push_to_hub(f"stair-lab/reeval_{args.task}-embed")
