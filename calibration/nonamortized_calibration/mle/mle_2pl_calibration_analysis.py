import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import DATASETS, error_bar_plot_single, goodness_of_fit_2PL_plot

if __name__ == "__main__":
    input_dir = "../data/mle_2pl_calibration"
    plot_dir = f"../plot/mle_2pl_calibration"
    os.makedirs(plot_dir, exist_ok=True)

    gof_means, gof_stds = [], []
    for dataset in tqdm(DATASETS):
        print(f"Processing {dataset}")
        y = pd.read_csv(
            f"../data/pre_calibration/{dataset}/matrix.csv", index_col=0
        ).values
        theta_hat = pd.read_csv(f"{input_dir}/{dataset}/theta.csv")["theta"].values
        z2_hat = pd.read_csv(f"{input_dir}/{dataset}/z.csv")["z2"].values
        z3_hat = pd.read_csv(f"{input_dir}/{dataset}/z.csv")["z3"].values

        gof_mean, gof_std = goodness_of_fit_2PL_plot(
            theta=torch.tensor(theta_hat, dtype=torch.float32),
            z2=torch.tensor(z2_hat, dtype=torch.float32),
            z3=torch.tensor(z3_hat, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            plot_path=f"{plot_dir}/goodness_of_fit_{dataset}",
        )
        gof_means.append(gof_mean)
        gof_stds.append(gof_std)

    gof_df = pd.DataFrame(
        {"datasets": DATASETS, "gof_means": gof_means, "gof_stds": gof_stds}
    )
    gof_df.to_csv(f"{plot_dir}/mle_2pl_calibration_gof.csv", index=False)

    error_bar_plot_single(
        datasets=DATASETS,
        means=gof_means,
        stds=gof_stds,
        plot_path=f"{plot_dir}/mle_2pl_calibration_summarize_gof",
        xlabel=r"Goodness of Fit",
    )
