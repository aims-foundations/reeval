from tqdm import tqdm
from datasets import Dataset, DatasetDict
import pandas as pd
import os
import json
from huggingface_hub import login
from dotenv import load_dotenv
from utils import DATASETS

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)

    input_dir = '../data/get_embed/'
    dataset_dict = {}
    for dataset in tqdm(DATASETS):
        with open(f'{input_dir}/{dataset}/embed.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            dataset_split = Dataset.from_pandas(df)
            dataset_dict[dataset] = dataset_split

    hf_dataset_dict = DatasetDict(dataset_dict)
    hf_dataset_dict.push_to_hub("stair-lab/reeval_individual-embed")
