import argparse
import subprocess
import wandb

if __name__ == "__main__":
    wandb.init(project="cat")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='semi_syn')
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    subprocess.run(f"conda run -n cat Rscript cat.R {args.task} {args.dataset}", shell=True, check=True)
