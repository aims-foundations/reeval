import pickle

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

dataset = "mmlu"
data_folder = snapshot_download(repo_id="stair-lab/reeval_csv", repo_type="dataset")
df_responses = pickle.load(open(f"{data_folder}/{dataset}/responses.pkl", "rb"))
instances = pd.read_csv(f"{data_folder}/{dataset}/instances.csv")

# merge df_responses and instances by instance_id
# instance_id in df_responses is float, while in instances is int
# but they are the same
df_responses["instance_id"] = df_responses["instance_id"].astype(int)
df = pd.merge(df_responses, instances, on="instance_id")

# get a list of unique subject
subjects = df["subject"].unique()

mean_score_per_subject = []
for i in range(len(subjects)):
    # filter out the row of subject[0] and subject[1]
    subjects_subset = subjects[0 + i : 1 + i]
    df_subject = df[df["subject"].isin(subjects_subset)]

    # only keep the model_id, instance_id, and exact_match columns
    df_subject = df_subject[["model_id", "instance_id", "exact_match"]]

    # create a pivot table with model_id as index, instance_id as columns, and exact_match as values
    df_subject = df_subject.pivot(
        index="model_id", columns="instance_id", values="exact_match"
    )

    # drop the columns with only 0 or 1 value
    df_subject = df_subject.loc[:, (df_subject != 0).any()]
    df_subject = df_subject.loc[:, (df_subject != 1).any()]

    # check if there is any nan value
    print(df_subject.isnull().values.any())

    # convert type to int
    df_subject = df_subject.astype(int)

    # count the number of row
    print(df_subject.shape[0])

    # only append if the number of row is the same as 75
    if df_subject.shape[0] == 75:
        mean_score_per_subject.append(df_subject.mean(1))

mean_score_per_subject = np.stack(mean_score_per_subject)

import matplotlib.pyplot as plt

for i in range(75):
    # color gradient from blue to red
    plt.plot(
        range(56),
        mean_score_per_subject[:, i],
        color=(i / 75, 0, 1 - i / 75),
        alpha=0.5,
    )

plt.savefig("ability_across_task.png", dpi=300, bbox_inches="tight")

breakpoint()

# save the data to csv file without index and header and type int
df_subject.to_csv("Subset.csv", index=False, header=False)

# Run the mirt.R script to get the item_difficulty and ability_MAP
import subprocess

subprocess.run(["Rscript", "mirt.R"])

# load item_difficulty and ability_MAP csv into a torch tensor
item_difficulty = pd.read_csv("item_difficulty.csv", header=None)
ability = pd.read_csv("ability_MAP.csv", header=None)
item_difficulty = torch.tensor(item_difficulty.values)
item_difficulty = item_difficulty[:, 1]
ability = torch.tensor(ability.values)[:, 0]

n_item = item_difficulty.shape[0]
n_testtaker = ability.shape[0]

item_difficulty = item_difficulty[None, :].repeat(ability.shape[0], 1)
ability = ability[:, None].repeat(1, item_difficulty.shape[1])
prob_correct = torch.sigmoid(ability - item_difficulty).numpy()

# compute AUC ROC with prob = prob_correct and ground truth = df_subject
from sklearn.metrics import roc_auc_score

y_true = df_subject.values.flatten()
y_score = prob_correct.flatten()
print("AUC ROC:", roc_auc_score(y_true, y_score))
print("Brier score:", np.mean((y_true - y_score) ** 2))
