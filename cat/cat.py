import argparse
import os
import subprocess

import wandb

if __name__ == "__main__":
    wandb.init(project="cat")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    args = parser.parse_args()

    output_dir = f"../result/cat_result/{args.scenario}"
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        f"conda run -n cat Rscript cat.R {args.scenario}", shell=True, check=True
    )
