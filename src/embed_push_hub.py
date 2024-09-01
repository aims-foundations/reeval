from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv
         
if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)

    search_df = pd.read_csv('/Users/tyhhh/Desktop/certified-eval/data/real/response_matrix/insex_search.csv')
    Z_path = "/Users/tyhhh/Desktop/certified-eval/data/real/irt_result/Z/all_1PL_Z_clean.csv"
    
    question_df = search_df.iloc[:, [1, 2]]  # Second column for question text, third column for perturb label
    Z_df = pd.read_csv(Z_path, usecols=[2])

    print(f"len(question_df): {len(question_df)}")
    print(f"len(Z_df): {len(Z_df)}")
    
    same_value_columns_indices = search_df[search_df.iloc[:, -1] == 1].index.tolist()
    print(f"len(same_value_columns_indices): {len(same_value_columns_indices)}")
    
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

    dataset_dict.push_to_hub("stair-lab/airbench-difficulty", private=True)