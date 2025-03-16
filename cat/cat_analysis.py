import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.iclr2024())
plt.style.use("seaborn-v0_8-paper")

def plot_cat(
    randoms,
    cats,
    cat_subs,
    plot_path,
    ylabel,
    hline_value=None,
    show_value=True,
):
    plt.figure(figsize=(6, 6))
    plt.plot(randoms, label="Random", color="red", linewidth=2)
    plt.plot(cats, label="Fisher large", color="blue", linewidth=2)
    plt.plot(cat_subs, label="Fisher small", color="darkgoldenrod", linewidth=2)
    plt.tick_params(axis="both", labelsize=25)
    plt.ylabel(ylabel, fontsize=25)
    if hline_value:
        plt.axhline(y=hline_value, color="black", linestyle="--", linewidth=2)
        if show_value:
            ax = plt.gca()
            ax.text(
                -0.03, hline_value, f"{hline_value:.2f}", transform=ax.get_yaxis_transform(),
                va="center", ha="right", color="black", fontsize=15
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
    plot_std=True,
    average_line=False,
):
    sorted_data = sorted(
        zip(datasets, means_train, stds_train, means_test, stds_test),
        key=lambda x: x[0],
    )
    datasets, means_train, stds_train, means_test, stds_test = zip(*sorted_data)
    fig, ax = plt.subplots(figsize=(8, 18))

    if plot_std:
        stds_train_mul3 = [s * 3 for s in stds_train]
        stds_test_mul3 = [s * 3 for s in stds_test]
        ax.barh(
            datasets,
            means_train,
            xerr=[np.zeros(len(datasets)), stds_train_mul3],
            capsize=5,
            color="blue",
            alpha=0.4,
            error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "blue"},
        )
        ax.barh(
            datasets,
            means_test,
            xerr=[np.zeros(len(datasets)), stds_test_mul3],
            capsize=5,
            color="orange",
            alpha=0.4,
            error_kw={"elinewidth": 2, "capthick": 2, "ecolor": "orange"},
        )
    else:
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

    if average_line:
        avg_train = np.mean(means_train)
        avg_test = np.mean(means_test)
        ax.axvline(avg_train, color="blue", linestyle="--", linewidth=2)
        ax.axvline(avg_test, color="orange", linestyle="--", linewidth=2)
        max_y = len(datasets) - 1
        ax.text(
            avg_train - 0.5,
            max_y,
            f"{avg_train:.2f}",
            color="blue",
            fontsize=25,
            ha="center",
        )
        ax.text(
            avg_test + 0.5,
            max_y,
            f"{avg_test:.2f}",
            color="orange",
            fontsize=25,
            ha="center",
        )

    ax.set_xlabel(xlabel, fontsize=35)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlim(0, xlim_upper)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
scenarios = [
    "babi_qa",
    "bbq",
    # "blimp",
    "boolq",
    "civil_comments",
    "commonsense",
    "dyck_language_np=3",
    "entity_data_imputation",
    "entity_matching",
    "gsm",
    "imdb",
    "legal_support",
    "legalbench",
    "math",
    "med_qa",
    "mmlu",
    "raft",
    "synthetic_reasoning",
    "thai_exam",
    "truthful_qa",
    "wikifact"
]

if __name__ == "__main__":
    plot_dir = f"cat_plot"
    os.makedirs(plot_dir, exist_ok=True)

    cat_reliability_95s, cat_mse_02s = [], []
    random_reliability_95s, random_mse_02s = [], []
    for scenario in tqdm(scenarios):
        input_path_sub = f"cat_result/{scenario}/cat_sub.csv"
        input_path_full = f"cat_result/{scenario}/cat_full.csv"
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
            hline_value=0.85
        )

        plot_cat(
            randoms=random_mse_list,
            cats=cat_mse_list,
            cat_subs=subset_cat_mse_list,
            plot_path=f"{plot_dir}/mse_{scenario}",
            ylabel=r"MSE",
            hline_value=0.4,
            show_value=False,
        )

        cat_reliability_95 = (
            min(
                [
                    i
                    for i in range(len(cat_reliability_list))
                    if cat_reliability_list[i] >= 0.85
                ],
                default=50,
            )
            + 1
        )
        cat_mse_02 = (
            min(
                [i for i in range(len(cat_mse_list)) if cat_mse_list[i] <= 0.4],
                default=50,
            )
            + 1
        )
        random_reliability_95 = (
            min(
                [
                    i
                    for i in range(len(random_reliability_list))
                    if random_reliability_list[i] >= 0.85
                ],
                default=50,
            )
            + 1
        )
        random_mse_02 = (
            min(
                [i for i in range(len(random_mse_list)) if random_mse_list[i] <= 0.4],
                default=50,
            )
            + 1
        )

        cat_reliability_95s.append(cat_reliability_95)
        cat_mse_02s.append(cat_mse_02)
        random_reliability_95s.append(random_reliability_95)
        random_mse_02s.append(random_mse_02)

    error_bar_plot_double(
        datasets=scenarios,
        means_train=random_reliability_95s,
        stds_train=[0] * len(scenarios),
        means_test=cat_reliability_95s,
        stds_test=[0] * len(scenarios),
        plot_path=f"{plot_dir}/cat_summarize_reliability",
        xlabel=r"Realiablity Reach 0.85",
        xlim_upper=50,
        plot_std=False,
    )

    error_bar_plot_double(
        datasets=scenarios,
        means_train=random_mse_02s,
        stds_train=[0] * len(scenarios),
        means_test=cat_mse_02s,
        stds_test=[0] * len(scenarios),
        plot_path=f"{plot_dir}/cat_summarize_mse",
        xlabel=r"MSE Reach 0.4",
        xlim_upper=50,
        plot_std=False,
    )
