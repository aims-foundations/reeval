import re
import argparse
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

scenario2pattern = {
    "mmlu/mmlu": r'(Question:)'
    # TODO: this pattern do not work for all scenarios, but split by '\n\n' is also problemetic, should write a scenario2pattern config
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True) # mmlu/mmlu
    args = parser.parse_args()
    
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_matrices", repo_type="dataset"
    )
    question_keys = pd.read_csv(f"{data_folder}/{args.dataset}/question_keys.csv")
    
    pattern = scenario2pattern[args.dataset]

    prompt_parts = question_keys['prompt'].apply(lambda x: re.split(pattern, x))
    prompt_parts = prompt_parts.apply(lambda x: [''.join(x[i:i+2]).strip() for i in range(1, len(x), 2)])
    num_split_parts = len(prompt_parts.iloc[0])
    assert all(len(x) == num_split_parts for x in prompt_parts), "Rows have inconsistent number of splits!"
    print(f"{args.dataset}, number of split parts: {num_split_parts}")
    question_keys['zero_shot'] = prompt_parts.str[-1]
    
    # question_keys.to_csv(f"{data_folder}/{args.dataset}/question_keys.csv", index=False)
    
    print(question_keys)