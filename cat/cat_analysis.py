import os

import pandas as pd
from tqdm import tqdm
from utils import DATASETS, error_bar_plot_double, plot_cat

if __name__ == "__main__":
    plot_dir = f"../plot/cat"
    os.makedirs(plot_dir, exist_ok=True)

    cat_reliability_95s, cat_mse_02s = [], []
    random_reliability_95s, random_mse_02s = [], []
    for dataset in tqdm(DATASETS):
        input_path_sub = f"../data/cat/{dataset}/cat_subset.csv"
        input_path_full = f"../data/cat/{dataset}/cat_full.csv"
        input_df_sub = pd.read_csv(input_path_sub)
        input_df_full = pd.read_csv(input_path_full)

        cat_data = input_df_full[input_df_full["variant"] == "CAT"]
        cat_reliability_list = cat_data["reliability"].tolist()
        cat_mse_list = cat_data["mse"].tolist()

        random_data = input_df_full[input_df_full["variant"] == "Random"]
        random_reliability_list = random_data["reliability"].tolist()
        random_mse_list = random_data["mse"].tolist()

        subset_cat_data = input_df_sub[input_df_sub["variant"] == "CAT"]
        subset_cat_reliability_list = subset_cat_data["reliability"].tolist()
        subset_cat_mse_list = subset_cat_data["mse"].tolist()

        plot_cat(
            randoms=random_reliability_list,
            cats=cat_reliability_list,
            cat_subs=subset_cat_reliability_list,
            plot_path=f"{plot_dir}/reliability_{dataset}",
            ylabel=r"Reliability",
        )

        plot_cat(
            randoms=random_mse_list,
            cats=cat_mse_list,
            cat_subs=subset_cat_mse_list,
            plot_path=f"{plot_dir}/mse_{dataset}",
            ylabel=r"MSE",
        )

        cat_reliability_95 = (
            min(
                [
                    i
                    for i in range(len(cat_reliability_list))
                    if cat_reliability_list[i] >= 0.95
                ],
                default=400,
            )
            + 1
        )
        cat_mse_02 = (
            min(
                [i for i in range(len(cat_mse_list)) if cat_mse_list[i] <= 0.2],
                default=400,
            )
            + 1
        )
        random_reliability_95 = (
            min(
                [
                    i
                    for i in range(len(random_reliability_list))
                    if random_reliability_list[i] >= 0.95
                ],
                default=400,
            )
            + 1
        )
        random_mse_02 = (
            min(
                [i for i in range(len(random_mse_list)) if random_mse_list[i] <= 0.2],
                default=400,
            )
            + 1
        )

        cat_reliability_95s.append(cat_reliability_95)
        cat_mse_02s.append(cat_mse_02)
        random_reliability_95s.append(random_reliability_95)
        random_mse_02s.append(random_mse_02)

    error_bar_plot_double(
        datasets=DATASETS,
        means_train=random_reliability_95s,
        stds_train=[0] * len(DATASETS),
        means_test=cat_reliability_95s,
        stds_test=[0] * len(DATASETS),
        plot_path=f"{plot_dir}/cat_summarize_reliability_95",
        xlabel=r"Realiablity Reach 0.95",
        xlim_upper=400,
        plot_std=False,
    )

    error_bar_plot_double(
        datasets=DATASETS,
        means_train=random_mse_02s,
        stds_train=[0] * len(DATASETS),
        means_test=cat_mse_02s,
        stds_test=[0] * len(DATASETS),
        plot_path=f"{plot_dir}/cat_summarize_mse_95",
        xlabel=r"MSE Reach 0.2",
        xlim_upper=400,
        plot_std=False,
    )
