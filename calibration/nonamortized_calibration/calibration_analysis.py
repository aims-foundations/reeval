import argparse
import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm
from utils.constants import DATASETS
from utils.utils import (
    accuracy_plot,
    error_bar_plot_single,
    goodness_of_fit,
    goodness_of_fit_plot,
    str2bool,
    theta_corr_plot,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument(
        "--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"]
    )
    parser.add_argument("--amortized", type=str2bool, default=False)
    args = parser.parse_args()

    input_dir = f"../../data/{args.fitting_method}_{args.PL}pl{'_amortized' if args.amortized else ''}_calibration"
    plot_dir = f"../../plot/{args.fitting_method}_{args.PL}pl{'_amortized' if args.amortized else ''}_calibration"
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gof_means, gof_stds = [], []
    corr_ctt_means, corr_ctt_stds = [], []
    corr_helm_means, corr_helm_stds = [], []
    plugin_gof_train_means, plugin_gof_test_means = [], []
    amor_gof_train_means, amor_gof_test_means = [], []

    for dataset in tqdm(DATASETS):
        print(f"Processing {dataset}")
        y = pd.read_csv(
            f"../../data/pre_calibration/{dataset}/matrix.csv", index_col=0
        ).values
        y = torch.tensor(y, device=device).float()

        abilities = pickle.load(open(f"{input_dir}/{dataset}/abilities.pkl", "rb"))
        abilities = torch.tensor(abilities, device=device)

        item_parms = pickle.load(open(f"{input_dir}/{dataset}/item_parms.pkl", "rb"))
        item_parms = torch.tensor(item_parms, device=device)

        # metric 1: GOF
        gof_mean, gof_std = goodness_of_fit_plot(
            z=item_parms,
            theta=abilities,
            y=y,
            plot_path=f"{plot_dir}/goodness_of_fit_{dataset}",
        )
        gof_means.append(gof_mean)
        gof_stds.append(gof_std)

        # metric 2: correlation with CTT
        corr_ctt_mean, corr_ctt_std = theta_corr_plot(
            mode="ctt",
            theta=abilities,
            y=y,
            plot_path=f"{plot_dir}/theta_corr_ctt_{dataset}",
        )
        corr_ctt_means.append(corr_ctt_mean)
        corr_ctt_stds.append(corr_ctt_std)

        # metric 3: correlation with HELM
        corr_helm_mean, corr_helm_std = theta_corr_plot(
            mode="helm",
            data_folder="../../data",
            theta=abilities,
            dataset=dataset,
            plot_path=f"{plot_dir}/theta_corr_helm_{dataset}",
        )
        corr_helm_means.append(corr_helm_mean)
        corr_helm_stds.append(corr_helm_std)

        # metric 4: Accuracy
        acc_mean, acc_std = accuracy_plot(
            item_parms=item_parms,
            theta=abilities,
            y=y,
            plot_path=f"{plot_dir}/accuracy_{dataset}",
        )

    #     plugin_train_indices = pd.read_csv(
    #         f"../../data/plugin_regression/{dataset}/train_0.csv"
    #     )["index"].values
    #     plugin_test_indices = pd.read_csv(
    #         f"../../data/plugin_regression/{dataset}/test_0.csv"
    #     )["index"].values

    #     plugin_gof_train_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[plugin_train_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, plugin_train_indices], dtype=torch.float32),
    #     )
    #     plugin_gof_train_means.append(plugin_gof_train_mean)

    #     plugin_gof_test_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[plugin_test_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, plugin_test_indices], dtype=torch.float32),
    #     )
    #     plugin_gof_test_means.append(plugin_gof_test_mean)

    #     amor_train_indices = pd.read_csv(
    #         f"../../data/amor_calibration/{dataset}/z_train_0.csv"
    #     )["index"].values
    #     amor_test_indices = pd.read_csv(
    #         f"../../data/amor_calibration/{dataset}/z_test_0.csv"
    #     )["index"].values

    #     amor_gof_train_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[amor_train_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, amor_train_indices], dtype=torch.float32),
    #     )
    #     amor_gof_train_means.append(amor_gof_train_mean)

    #     amor_gof_test_mean, _ = goodness_of_fit(
    #         z=torch.tensor(item_parms[amor_test_indices], dtype=torch.float32),
    #         theta=torch.tensor(abilities, dtype=torch.float32),
    #         y=torch.tensor(y[:, amor_test_indices], dtype=torch.float32),
    #     )
    #     amor_gof_test_means.append(amor_gof_test_mean)

    # plugin_gof_df_train = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": plugin_gof_train_means,
    #     }
    # )
    # plugin_gof_df_train.to_csv(f"{plot_dir}/nonamor4plugin_gof_train.csv", index=False)

    # plugin_gof_df_test = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": plugin_gof_test_means,
    #     }
    # )
    # plugin_gof_df_test.to_csv(f"{plot_dir}/nonamor4plugin_gof_test.csv", index=False)

    # amor_gof_df_train = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": amor_gof_train_means,
    #     }
    # )
    # amor_gof_df_train.to_csv(f"{plot_dir}/nonamor4amor_gof_train.csv", index=False)

    # amor_gof_df_test = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "gof_means": amor_gof_test_means,
    #     }
    # )
    # amor_gof_df_test.to_csv(f"{plot_dir}/nonamor4amor_gof_test.csv", index=False)

    # gof_df = pd.DataFrame(
    #     {"datasets": DATASETS, "gof_means": gof_means, "gof_stds": gof_stds}
    # )
    # gof_df.to_csv(f"{plot_dir}/nonamor_calibration_gof.csv", index=False)

    # ctt_df = pd.DataFrame(
    #     {
    #         "datasets": DATASETS,
    #         "corr_ctt_means": corr_ctt_means,
    #         "corr_ctt_stds": corr_ctt_stds,
    #     }
    # )
    # ctt_df.to_csv(f"{plot_dir}/nonamor_calibration_corr_ctt.csv", index=False)

    # helm_df = pd.DataFrame(
    #     {
    #         "datasets": [d for d in DATASETS if d != "airbench"],
    #         "corr_helm_means": corr_helm_means,
    #         "corr_helm_stds": corr_helm_stds,
    #     }
    # )
    # helm_df.to_csv(f"{plot_dir}/nonamor_calibration_corr_helm.csv", index=False)

    # error_bar_plot_single(
    #     datasets=DATASETS,
    #     means=gof_means,
    #     stds=gof_stds,
    #     plot_path=f"{plot_dir}/nonamor_calibration_summarize_gof",
    #     xlabel=r"Goodness of Fit",
    # )

    # error_bar_plot_single(
    #     datasets=DATASETS,
    #     means=corr_ctt_means,
    #     stds=corr_ctt_stds,
    #     plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_ctt",
    #     xlabel=r"$\theta$ correlation with CTT",
    # )

    # error_bar_plot_single(
    #     datasets=[d for d in DATASETS if d != "airbench"],
    #     means=corr_helm_means,
    #     stds=corr_helm_stds,
    #     plot_path=f"{plot_dir}/nonamor_calibration_summarize_theta_corr_helm",
    #     xlabel=r"$\theta$ correlation with HELM",
    # )
