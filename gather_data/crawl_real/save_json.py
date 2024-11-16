import argparse
import os

import pandas as pd
import requests
import wandb

if __name__ == "__main__":
    wandb.init(project="save_json")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard", type=str, default="classic", choices=["classic", "mmlu"]
    )
    parser.add_argument("--dataset", type=str, required=True)  # use wandb sweep, mmlu
    args = parser.parse_args()

    output_dir = f"../../../data/gather_data/crawl_real/jsons/{args.dataset}_json"
    os.makedirs(output_dir, exist_ok=True)

    full_strings_all = pd.read_csv(
        f"../../../data/gather_data/crawl_real/crawl_dataset_name_{args.leaderboard}.csv"
    )["Run"].tolist()
    full_strings = [
        f for f in full_strings_all if (f.split(":")[0].split(",")[0] == args.dataset)
    ]
    for full_string in full_strings:
        save_path = f"{output_dir}/{full_string}.json"
        if os.path.exists(save_path):
            continue

        if args.leaderboard == "classic":
            base_url = "https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0."
            max_version = 4
        elif args.leaderboard == "mmlu":
            base_url = "https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/runs/v1."
            max_version = 8

        found_tag = False
        for i in range(max_version + 1):
            url = f"{base_url}{i}.0/{full_string}/scenario_state.json"
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, "wb") as file:
                    file.write(response.content)
                found_tag = True
                break
        if found_tag == False:
            print(
                f"Failed to download the file for {full_string}. Status code:",
                response.status_code,
            )
