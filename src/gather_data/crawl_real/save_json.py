import argparse
import pandas as pd
import requests
import os
import wandb

if __name__ == "__main__":
    wandb.init(mode = "offline")
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=str, required=True) # classic, mmlu
    parser.add_argument('--start_string', type=str, required=True) # use wandb sweep, mmlu
    args = parser.parse_args()
  
    full_strings_all = pd.read_csv(f'../../../data/real/crawl/crawl_dataset_name_{args.leaderboard}.csv')['Run'].tolist()
    
    save_dir = f'../../../data/real/crawl/{args.start_string}_json'
    os.makedirs(save_dir, exist_ok=True)
    full_strings = [f for f in full_strings_all if f.startswith(args.start_string)]
    for full_string in full_strings:
        save_path = f'{save_dir}/{full_string}.json'
        if os.path.exists(save_path):
            continue
        
        if args.leaderboard == "classic":
            base_url = 'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.'
            max_version = 4
        elif args.leaderboard == "mmlu":
            base_url = 'https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/runs/v1.'
            max_version = 8
        
        for i in range(max_version + 1):
            url = f'{base_url}{i}.0/{full_string}/scenario_state.json'
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                break
    
            print(f'Failed to download the file for {full_string}. Status code:', response.status_code)
