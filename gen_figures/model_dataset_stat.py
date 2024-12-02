import os

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm
from utils.constants import DATASETS, PLOT_NAME_MAP

if __name__ == "__main__":
    DATASETS = [d for d in DATASETS if d != "combined_data"]
    DATASETS = sorted(DATASETS)
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )
    model_names = []
    for dataset in tqdm(DATASETS):
        model_name = pd.read_csv(
            f"{data_folder}/{dataset}/matrix.csv", index_col=0
        ).index.tolist()
        model_names.extend(model_name)
    model_names = sorted(list(set(model_names)))

    # create a dataframe, each row is a model, each column is a dataset,
    # the value is 1 if the model is in the dataset
    df = pd.DataFrame(0, index=model_names, columns=DATASETS)
    for dataset in DATASETS:
        model_name = pd.read_csv(
            f"{data_folder}/{dataset}/matrix.csv", index_col=0
        ).index.tolist()
        df.loc[model_name, dataset] = 1
    df.columns = [PLOT_NAME_MAP.get(col, col) for col in df.columns]

    output_dir = "../plot/model_dataset_stat"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/model_dataset_stat.csv")

    data_transposed = df.T
    plt.figure(figsize=(40, 30))
    plt.imshow(data_transposed, cmap="Blues", interpolation="nearest")
    plt.xticks(
        range(data_transposed.shape[1]),
        data_transposed.columns,
        rotation=45,
        ha="right",
        fontsize=6,
    )
    plt.yticks(range(data_transposed.shape[0]), data_transposed.index, fontsize=6)
    plt.grid(visible=False)

    # plt.gca().set_xticks([x - 0.5 for x in range(1, data_transposed.shape[1])], minor=True)
    # plt.gca().set_yticks([y - 0.5 for y in range(1, data_transposed.shape[0])], minor=True)
    # plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    for x in range(data_transposed.shape[1] + 1):
        plt.axvline(x - 0.5, color="gray", linestyle="--", linewidth=0.5)
    for y in range(data_transposed.shape[0] + 1):
        plt.axhline(y - 0.5, color="gray", linestyle="--", linewidth=0.5)

    plt.savefig(f"{output_dir}/model_dataset_stat.png", dpi=300, bbox_inches="tight")
    plt.close()
