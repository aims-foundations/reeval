"""
Plotting number of test takers and questions in each benchmark,
which is the Figure 2 in the paper.
"""

import os

import pandas as pd

from gen_figures.plot import plot_bar
from huggingface_hub import snapshot_download
from utils.constants import DATASETS, PLOT_NAME_MAP


if __name__ == "__main__":
    DATASETS = [d for d in DATASETS if d != "combined_data"]
    plot_dir = f"../plot/others"
    os.makedirs(plot_dir, exist_ok=True)
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    question_nums, testtaker_nums = [], []
    for dataset in DATASETS:
        y = pd.read_csv(f"{data_folder}/{dataset}/matrix.csv", index_col=0).values
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
