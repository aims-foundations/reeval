import pandas as pd
import requests
from tqdm import tqdm

if __name__ == "__main__":
    file_path = f'../../data/real/crawl/crawl_dataset_name_classic.csv'
    df = pd.read_csv(file_path)
    
    start_strings = [
        "synthetic_reasoning:mode=induction", 
        "synthetic_reasoning:mode=pattern_match",
        "synthetic_reasoning:mode=variable_substitution",
        "synthetic_reasoning_natural:difficulty=easy",
        "synthetic_reasoning_natural:difficulty=hard"
    ]
    
    for start_string in start_strings:
        print(start_string)
        filtered_df = df[df['Run'].str.startswith(start_string)]
        
        for i, row in tqdm(filtered_df.iterrows()):
            exp_string = row['Run']
            base_url = 'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.'
            max_version = 4
            version_found = False
            
            for i in range(max_version + 1):
                url = f'{base_url}{i}.0/{exp_string}/scenario_state.json'
                response = requests.get(url)
                if response.status_code == 200:
                    version_found = True
                    file_name = f'../../data/real/crawl/synthetic_reasoning_json/{exp_string}.json'
                    with open(file_name, 'wb') as file:
                        file.write(response.content)
               
            if not version_found:
                print(f'Failed to download the file for {exp_string}. Status code:', response.status_code)