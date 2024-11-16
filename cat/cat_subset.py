import argparse
import os
import subprocess

import wandb

if __name__ == "__main__":
    wandb.init(project="cat")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="semi_syn")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    output_dir = f"../data/cat_airbench/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        f"conda run -n cat Rscript cat_airbench.R {args.task} {args.dataset}",
        shell=True,
        check=True,
    )
