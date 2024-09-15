import argparse
import pandas as pd
import requests
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    args = parser.parse_args()
    exp = args.exp
    
    file_path = f'../../data/real/crawl/crawl_dataset_name_{exp}.csv'
    df = pd.read_csv(file_path)
    
    if exp == "synthetic_reasoning":
        start_strings = [
            "synthetic_reasoning:mode=induction", 
            "synthetic_reasoning:mode=pattern_match",
            "synthetic_reasoning:mode=variable_substitution",
            "synthetic_reasoning_natural:difficulty=easy",
            "synthetic_reasoning_natural:difficulty=hard"
        ]
    elif exp == "mmlu":
        start_strings = pd.read_csv('../../data/real/crawl/dataset_info_stats_mmlu.csv')['dataset_name'].tolist()
    
    for start_string in tqdm(start_strings):
        if exp == "synthetic_reasoning":
            filtered_df = df[df['Run'].str.startswith(start_string)]
        elif exp == "mmlu":
            start_string_2 = start_string.split(",eval_split")[0]
            filtered_df = df[df['Run'].str.startswith(start_string_2)]
        
        for i, row in filtered_df.iterrows():
            exp_string = row['Run']
            if exp == "synthetic_reasoning":
                base_url = 'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.'
                max_version = 4
            elif exp == "mmlu":
                base_url = 'https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/runs/v1.'
                max_version = 8
            version_found = False
            
            for i in range(max_version + 1):
                url = f'{base_url}{i}.0/{exp_string}/scenario_state.json'
                response = requests.get(url)
                if response.status_code == 200:
                    if exp == "synthetic_reasoning":
                        file_name = f'../../data/real/crawl/synthetic_reasoning_json/{exp_string}.json'
                    elif exp == "mmlu":
                        file_name = f'../../data/real/crawl/mmlu_json/{exp_string}.json'
                        
                    with open(file_name, 'wb') as file:
                        file.write(response.content)
                        
                    version_found = True
                    break
               
            if not version_found:
                print(f'Failed to download the file for {exp_string}. Status code:', response.status_code)