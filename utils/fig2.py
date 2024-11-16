import os

import pandas as pd

from utils import DATASETS, plot_bar

if __name__ == "__main__":
    plot_dir = f"../plot/others"
    os.makedirs(plot_dir, exist_ok=True)

    question_nums, testtaker_nums = [], []
    for dataset in DATASETS:
        y = pd.read_csv(
            f"../data/pre_calibration/{dataset}/matrix.csv", index_col=0
        ).values
        testtaker_nums.append(y.shape[0])
        question_nums.append(y.shape[1])

    plot_bar(
        DATASETS,
        question_nums,
        f"{plot_dir}/question_nums.png",
        r"Number of Questions",
        exp_axis=True,
    )
    plot_bar(
        DATASETS,
        testtaker_nums,
        f"{plot_dir}/testtaker_nums.png",
        r"Number of Test Takers",
    )
