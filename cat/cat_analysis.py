import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.iclr2024())


def plot_cat(
    randoms,
    cats,
    cat_subs,
    plot_path,
    ylabel,
    hline_value=None,
    show_value=True,
):
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        plt.figure(figsize=(10, 10))
        plt.plot(randoms, label="Random", color="red", linewidth=2)
        plt.plot(cats, label=r"Fisher (Full Bank)", color="blue", linewidth=2)
        plt.plot(
            cat_subs, label=r"Fisher (10\% Bank)", color="darkgoldenrod", linewidth=2
        )
        plt.tick_params(axis="both", labelsize=25)
        plt.ylabel(ylabel, fontsize=25)
        if hline_value:
            plt.axhline(y=hline_value, color="black", linestyle="--", linewidth=2)
            if show_value:
                ax = plt.gca()
                ax.text(
                    -0.03,
                    hline_value,
                    f"{hline_value:.2f}",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="right",
                    color="black",
                    fontsize=15,
                )
        plt.ylim(0, 1)
        plt.legend(fontsize=25)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()


def error_bar_plot_double(
    datasets,
    means_train,
    stds_train,
    means_test,
    stds_test,
    plot_path,
    xlabel,
    xlim_upper=1.1,
):
    sorted_data = sorted(
        zip(datasets, means_train, stds_train, means_test, stds_test),
        key=lambda x: x[3],
    )
    datasets, means_train, stds_train, means_test, stds_test = zip(*sorted_data)
    fig, ax = plt.subplots(figsize=(8, 18))

    ax.barh(datasets, means_train, color="blue", alpha=0.4)
    ax.barh(datasets, means_test, color="orange", alpha=0.4)
    print("")
    print(xlabel)
    improvements = []
    for dataset, mse_train, mse_test in zip(datasets, means_train, means_test):
        improvement = (mse_train - mse_test) / mse_train
        improvements.append((dataset, improvement))

    improvements.sort(key=lambda x: x[1], reverse=True)
    # print mean improvement
    print(
        f"Mean improvement: {np.mean([improvement for _, improvement in improvements])}"
    )
    for dataset, improvement in improvements:
        print(f"{dataset}: {improvement}")

    ax.set_xlabel(xlabel, fontsize=35)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlim(0, xlim_upper)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


scenarios = [
    "air_bench_2024",
    "babi_qa",
    "bbq",
    "boolq",
    "civil_comments",
    "commonsense",
    "dyck_language_np=3",
    # "entity_data_imputation",
    "entity_matching",
    "gsm",
    "imdb",
    "legal_support",
    "legalbench",
    "lsat_qa",
    "math",
    "med_qa",
    "mmlu",
    "raft",
    "synthetic_reasoning",
    "thai_exam",
    "truthful_qa",
    "wikifact",
]

if __name__ == "__main__":
    plot_dir = "../result/cat_plot"
    os.makedirs(plot_dir, exist_ok=True)

    cat_reli_reachs, cat_mse_reachs = [], []
    random_reli_reachs, random_mse_reachs = [], []
    for scenario in tqdm(scenarios):
        input_path_sub = f"../result/cat_result/{scenario}/cat_sub.csv"
        input_path_full = f"../result/cat_result/{scenario}/cat_full.csv"
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
            plot_path=f"{plot_dir}/reliability_{scenario}",
            ylabel=r"Reliability",
            hline_value=0.95,
            show_value=True,
        )

        plot_cat(
            randoms=random_mse_list,
            cats=cat_mse_list,
            cat_subs=subset_cat_mse_list,
            plot_path=f"{plot_dir}/mse_{scenario}",
            ylabel=r"MSE",
            hline_value=0.4,
            show_value=True,
        )

        cat_reli_reach = (
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
        cat_mse_reach = (
            min(
                [i for i in range(len(cat_mse_list)) if cat_mse_list[i] <= 0.4],
                default=400,
            )
            + 1
        )
        random_reli_reach = (
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
        random_mse_reach = (
            min(
                [i for i in range(len(random_mse_list)) if random_mse_list[i] <= 0.4],
                default=400,
            )
            + 1
        )

        cat_reli_reachs.append(cat_reli_reach)
        cat_mse_reachs.append(cat_mse_reach)
        random_reli_reachs.append(random_reli_reach)
        random_mse_reachs.append(random_mse_reach)

    error_bar_plot_double(
        datasets=scenarios,
        means_train=random_reli_reachs,
        stds_train=[0] * len(scenarios),
        means_test=cat_reli_reachs,
        stds_test=[0] * len(scenarios),
        plot_path=f"{plot_dir}/cat_summarize_reliability",
        xlabel=r"Realiablity Reach 0.95",
        xlim_upper=400,
    )

    error_bar_plot_double(
        datasets=scenarios,
        means_train=random_mse_reachs,
        stds_train=[0] * len(scenarios),
        means_test=cat_mse_reachs,
        stds_test=[0] * len(scenarios),
        plot_path=f"{plot_dir}/cat_summarize_mse",
        xlabel=r"MSE Reach 0.4",
        xlim_upper=400,
    )
