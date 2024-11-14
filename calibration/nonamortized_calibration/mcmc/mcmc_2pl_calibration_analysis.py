import os

import pandas as pd
from tqdm import tqdm
from utils import (
    DATASETS,
    error_bar_plot_single,
    theta_corr_ctt_plot,
    theta_corr_helm_plot,
)

if __name__ == "__main__":
    DATASETS = [d for d in DATASETS if d != "civil_comments" and d != "imdb"]
    input_dir = "../data/mcmc_2pl_calibration"
    plot_dir = f"../plot/mcmc_2pl_calibration"
    os.makedirs(plot_dir, exist_ok=True)

    gof_data = {
        "airbench": [0.92, 0.0],
        "twitter_aae": [0.91, 0.0],
        "math": [0.94, 0.0],
        "entity_data_imputation": [0.83, 0.0],
        "real_toxicity_prompts": [0.9, 0.0],
        # 'civil_comments': [0.0, 0.0],
        # 'imdb': [0.0, 0.0],
        "boolq": [0.79, 0.0],
        "wikifact": [0.85, 0.0],
        "babi_qa": [0.79, 0.0],
        "mmlu": [0.94, 0.0],
        "truthful_qa": [0.78, 0.0],
        "legal_support": [0.84, 0.0],
        "synthetic_reasoning": [0.92, 0.0],
        "quac": [0.95, 0.0],
        "entity_matching": [0.83, 0.0],
        "synthetic_reasoning_natural": [0.89, 0.0],
        "bbq": [0.78, 0.0],
        "raft": [0.8, 0.0],
        "narrative_qa": [0.58, 0.0],
        "commonsense": [0.78, 0.0],
        "lsat_qa": [0.89, 0.0],
        "bold": [0.92, 0.0],
        "dyck_language_np3": [0.92, 0.0],
    }
    gof_means = [gof_data[dataset][0] for dataset in DATASETS]
    gof_stds = [gof_data[dataset][1] for dataset in DATASETS]

    corr_ctt_means, corr_ctt_stds = [], []
    corr_helm_means, corr_helm_stds = [], []
    for dataset in tqdm(DATASETS):
        print(f"Processing {dataset}")
        y = pd.read_csv(
            f"../data/pre_calibration/{dataset}/matrix.csv", index_col=0
        ).values
        theta_hat = pd.read_csv(f"{input_dir}/{dataset}/theta.csv")["theta"].values
        # z_hat = pd.read_csv(f'{input_dir}/{dataset}/z.csv')['z'].values

        corr_ctt_mean, corr_ctt_std = theta_corr_ctt_plot(
            theta=theta_hat,
            y=y,
            plot_path=f"{plot_dir}/theta_corr_ctt_{dataset}",
        )
        corr_ctt_means.append(corr_ctt_mean)
        corr_ctt_stds.append(corr_ctt_std)

        if dataset != "airbench":
            corr_helm_mean, corr_helm_std = theta_corr_helm_plot(
                theta=theta_hat,
                dataset=dataset,
                plot_path=f"{plot_dir}/theta_corr_helm_{dataset}",
            )
            corr_helm_means.append(corr_helm_mean)
            corr_helm_stds.append(corr_helm_std)

    ctt_df = pd.DataFrame(
        {
            "datasets": DATASETS,
            "corr_ctt_means": corr_ctt_means,
            "corr_ctt_stds": corr_ctt_stds,
        }
    )
    ctt_df.to_csv(f"{plot_dir}/mcmc_2pl_calibration_corr_ctt.csv", index=False)

    helm_df = pd.DataFrame(
        {
            "datasets": [d for d in DATASETS if d != "airbench"],
            "corr_helm_means": corr_helm_means,
            "corr_helm_stds": corr_helm_stds,
        }
    )
    helm_df.to_csv(f"{plot_dir}/mcmc_2pl_calibration_corr_helm.csv", index=False)

    error_bar_plot_single(
        datasets=DATASETS,
        means=gof_means,
        stds=gof_stds,
        plot_path=f"{plot_dir}/mcmc_2pl_calibration_summarize_gof",
        xlabel=r"Goodness of Fit",
    )

    error_bar_plot_single(
        datasets=DATASETS,
        means=corr_ctt_means,
        stds=corr_ctt_stds,
        plot_path=f"{plot_dir}/mcmc_2pl_calibration_summarize_theta_corr_ctt",
        xlabel=r"$\theta$ correlation with CTT",
    )

    error_bar_plot_single(
        datasets=[d for d in DATASETS if d != "airbench"],
        means=corr_helm_means,
        stds=corr_helm_stds,
        plot_path=f"{plot_dir}/mcmc_2pl_calibration_summarize_theta_corr_helm",
        xlabel=r"$\theta$ correlation with HELM",
    )
