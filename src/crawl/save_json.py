import argparse
import pandas as pd
import requests
from tqdm import tqdm
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    args = parser.parse_args()
  
    full_strings_all = pd.read_csv(f'../../data/real/crawl/crawl_dataset_name_{args.exp}.csv')['Run'].tolist()
    stats_strings = pd.read_csv(f'../../data/real/crawl/dataset_info_stats_{args.exp}.csv')['dataset_name'].tolist()
    start_strings = set([s.split(":")[0] for s in stats_strings])
    
    for start_string in tqdm(start_strings):
        save_path = f'../../data/real/crawl/{start_string}_json'
        os.makedirs(save_path, exist_ok=True)
        full_strings = [f for f in full_strings_all if f.startswith(start_string)]
        for full_string in full_strings:
            if args.exp == "classic":
                base_url = 'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.'
                max_version = 4
            elif args.exp == "mmlu":
                base_url = 'https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/runs/v1.'
                max_version = 8
            version_found = False
            
            for i in range(max_version + 1):
                url = f'{base_url}{i}.0/{full_string}/scenario_state.json'
                response = requests.get(url)
                if response.status_code == 200:
                    with open(f'{save_path}/{full_string}.json', 'wb') as file:
                        file.write(response.content)
                    version_found = True
                    break
        
            if not version_found:
                print(f'Failed to download the file for {full_string}. Status code:', response.status_code)
