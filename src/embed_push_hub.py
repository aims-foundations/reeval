import argparse
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
       
def push_hub_with_perturb(search_path, Z_path, hf_repo):
    search_df = pd.read_csv(search_path)
    Z_df = pd.read_csv(Z_path, usecols=["z3"])
    same_value_columns_indices = search_df[search_df.loc[:, "is_deleted"] == 1].index.tolist()
    
    question_df = search_df.loc[:, ["text", "perturb"]]
    question_df = question_df.drop(same_value_columns_indices).reset_index(drop=True)
    assert len(question_df) == len(Z_df), "The lengths of question_df and Z_df must be the same after filtering."
    
    combined_df = pd.concat([question_df, Z_df], axis=1)
    combined_df.columns = ['question_text', 'label', 'z3']
    
    base_df = combined_df[combined_df['label'] == 'base'].reset_index(drop=True)
    perturb1_df = combined_df[combined_df['label'] == 'perturb1'].reset_index(drop=True)
    perturb2_df = combined_df[combined_df['label'] == 'perturb2'].reset_index(drop=True)
    
    all_dataset = Dataset.from_pandas(combined_df.drop(columns=['label']))
    base_dataset = Dataset.from_pandas(base_df.drop(columns=['label']))
    perturb1_dataset = Dataset.from_pandas(perturb1_df.drop(columns=['label']))
    perturb2_dataset = Dataset.from_pandas(perturb2_df.drop(columns=['label']))
    
    dataset_dict = DatasetDict({
        "whole": all_dataset,
        "base": base_dataset,
        "perturb1": perturb1_dataset,
        "perturb2": perturb2_dataset
    })
    dataset_dict.push_to_hub(hf_repo, private=True)

def push_hub(search_path, Z_path, hf_repo):
    search_df = pd.read_csv(search_path)
    Z_df = pd.read_csv(Z_path, usecols=["z3"])
    same_value_columns_indices = search_df[search_df.loc[:, "is_deleted"] == 1].index.tolist()
    
    question_df = search_df.loc[:, ["text"]]
    question_df = question_df.drop(same_value_columns_indices).reset_index(drop=True)
    assert len(question_df) == len(Z_df), "The lengths of question_df and Z_df must be the same after filtering."
    
    combined_df = pd.concat([question_df, Z_df], axis=1)
    combined_df.columns = ['question_text', 'z3']
    
    all_dataset = Dataset.from_pandas(combined_df)
    dataset_dict = DatasetDict({
        "whole": all_dataset,
    })
    dataset_dict.push_to_hub(hf_repo, private=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='airbench')
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)

    if args.exp == 'airbench':
        push_hub_with_perturb(
            search_path='../data/real/response_matrix/normal/index_search.csv',
            Z_path='../data/real/irt_result/normal/Z/all_1PL_Z_clean.csv',
            hf_repo='stair-lab/airbench-difficulty'
        )
    
    elif args.exp == 'synthetic_reasoning':
        push_hub(
            search_path='../data/real/response_matrix/normal_syn_reason/mask_index_search.csv',
            Z_path='../data/real/irt_result/pyMLE_normal_syn_reason/Z/mask_1PL_Z.csv',
            hf_repo='stair-lab/synthetic_reasoning-difficulty'
        )
        
    