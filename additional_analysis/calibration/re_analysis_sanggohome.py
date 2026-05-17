import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from tueplots import bundles

bundles.icml2024()
import random

import matplotlib.colors as mcolors
import numpy as np
import torch
from torchmetrics import AUROC

auroc = AUROC(task="binary")
import pickle

from torch.distributions import Bernoulli

torch.manual_seed(0)
device = "cuda:3"
# device = "cpu"
import os

from huggingface_hub import snapshot_download
from torch.optim import LBFGS


def majority_vote(series):
    # Convert to NumPy array for fast vectorized operations
    arr = series.to_numpy()
    # Filter out -1 values (no value)
    valid = arr[arr != -1]
    if valid.size == 0:
        return -1
    # Use np.bincount to count occurrences of 0 and 1.
    # Ensure valid values are integers and count up to index 1.
    counts = np.bincount(valid.astype(int), minlength=2)
    count0, count1 = counts[0], counts[1]
    if count1 > count0:
        return 1
    elif count0 > count1:
        return 0
    else:
        return random.choice([0, 1])


def visualize_response_matrix(results, value, filename):
    # Extract the groups labels in the order of the columns
    group_values = results.columns.get_level_values("scenario")

    # Identify the boundaries where the group changes
    boundaries = []
    for i in range(1, len(group_values)):
        if group_values[i] != group_values[i - 1]:
            boundaries.append(i - 0.5)  # using 0.5 to place the line between columns

    # visualize the results with a matrix red is 0, white is -1 and blue is 1
    cmap = mcolors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Define boundaries for the three categories
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Calculate midpoints for each group label
    groups_list = list(group_values)
    group_names = []
    group_midpoints = []
    current_group = groups_list[0]
    start_index = 0
    for i, grp in enumerate(groups_list):
        if grp != current_group:
            midpoint = (start_index + i - 1) / 2.0
            group_names.append(current_group)
            group_midpoints.append(midpoint)
            current_group = grp
            start_index = i
    # Add the last group
    midpoint = (start_index + len(groups_list) - 1) / 2.0
    group_names.append(current_group)
    group_midpoints.append(midpoint)

    # Define the minimum spacing between labels (e.g., 500 units)
    min_spacing = 100
    last_label_pos = -float("inf")
    # Plot the matrix
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.matshow(value, aspect="auto", cmap=cmap, norm=norm)

        # Add vertical lines at each boundary
        for b in boundaries:
            ax.axvline(x=b, color="black", linewidth=0.25, linestyle="--", alpha=0.5)

        # Add group labels above the matrix, but only if they're at least 500 apart
        for name, pos in zip(group_names, group_midpoints):
            if pos - last_label_pos >= min_spacing:
                # name = eval(name)
                # name = "/".join(name)
                ax.text(
                    pos, -5, name, ha="center", va="bottom", rotation=90, fontsize=3
                )
                last_label_pos = pos

        # add model labels
        ax.set_yticks(range(len(results.index)))
        ax.set_yticklabels(results.index, fontsize=3)

        # Add colorbar
        cbar = plt.colorbar(cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["-1", "0", "1"])
        plt.savefig(f"{filename}.png", dpi=600, bbox_inches="tight")
        plt.close()


def trainer(parameters, optim, closure, verbose=True):
    pbar = tqdm(range(100)) if verbose else range(100)
    for iteration in pbar:
        if iteration > 0:
            # Clone each tensor individually for previous state
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()

        loss = optim.step(closure)

        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(
                torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None
            )
            if verbose:
                pbar.set_postfix(
                    {
                        "grad_norm": grad_norm,
                        "d_parameter": d_parameters,
                        "d_loss": d_loss,
                    }
                )

            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
    return parameters


def compute_auc(probs, data, train_idtor, test_idtor):
    train_probs = probs[train_idtor.bool()]
    test_probs = probs[test_idtor.bool()]
    train_labels = data[train_idtor.bool()]
    test_labels = data[test_idtor.bool()]

    print(f"train auc: {auroc(train_probs, train_labels)}")
    print(f"test auc: {auroc(test_probs, test_labels)}")


# results = pd.read_pickle("../gather_helm_data/helm_tables/responses.pkl")
# results_full = pd.read_pickle("results_perplexity.pkl")
# results_full = pd.read_pickle("results_perplexity_thirdattempt.pkl")
# results_full = pd.read_pickle("results_perplexity_forthattempt.pkl")
file_path = snapshot_download(
    repo_id="stair-lab/results_perplexity_forthattempt",
    repo_type="dataset",
    use_auth_token="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",
)
with open(f"{file_path}/results_perplexity_forthattempt.pkl", "rb") as f:
    results_full = pickle.load(f)

if os.path.exists("results.pkl"):
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

else:
    results_full = results_full.sample(frac=1).reset_index(drop=True)
    results = results_full[
        ["request.model", "request.prompt", "scenario", "dicho_score"]
    ]
    results = results.dropna(
        subset=["request.model", "request.prompt", "scenario", "dicho_score"]
    )
    # drop the dicho_score of 0.5
    results = results[results["dicho_score"] != 0.5]
    results["dicho_score"] = results["dicho_score"].astype(bool)
    assert results["dicho_score"].isin([0, 1]).all()
    results = results.drop_duplicates(
        subset=["request.model", "request.prompt", "scenario"], keep="first"
    )
    print(results.shape[0] / results_full.shape[0])

    # Count the number of unique request.prompt for each request.model
    model_prompt_counts = results.groupby("request.model", observed=True)[
        "request.prompt"
    ].nunique()
    # Count the number of unique request.model for each request.prompt
    prompt_model_counts = results.groupby("request.prompt", observed=True)[
        "request.model"
    ].nunique()
    # Identify models with at least 10 unique prompts and prompts with at least 10 unique models
    models_to_keep = model_prompt_counts[model_prompt_counts >= 30].index
    prompts_to_keep = prompt_model_counts[prompt_model_counts >= 30].index
    # Filter the DataFrame accordingly
    results = results[
        results["request.model"].isin(models_to_keep)
        & results["request.prompt"].isin(prompts_to_keep)
    ]

    results = results.pivot(
        index="request.model",
        columns=["request.prompt", "scenario"],
        values="dicho_score",
    )

    # sort the columns by groups
    results = results.sort_index(axis=1, level="scenario")

    results = results.loc[:, (results != 0).any()]
    results = results.loc[:, (results != 1).any()]
    results = results.fillna(-1).astype(int)
    # Replace -1 with NaN so that missing scores are ignored
    results = results.replace(-1, np.nan)

    # Compute the overall average for each group manually
    group_means = {}
    for group in results.columns.get_level_values("scenario").unique():
        mask = results.columns.get_level_values("scenario") == group
        values = results.loc[:, mask].values  # all values for this group
        group_means[group] = np.nanmean(values)

    # Sort the scenario by their average score
    sorted_groups = sorted(group_means, key=group_means.get)

    # Create a mapping from group to its sort order
    group_order = {group: order for order, group in enumerate(sorted_groups)}

    # Reorder the columns based on the new group order using the key parameter
    results = results.sort_index(
        axis=1, level="scenario", key=lambda x: x.map(group_order)
    )

    # Compute the overall average for each row (ignoring NaNs)
    row_means = results.mean(axis=1)

    # Sort the rows by these computed averages (lowest to highest)
    results = results.loc[row_means.sort_values().index]

    # convert nan back to -1
    results = results.replace(np.nan, -1)
    # count the fraction of -1
    print((results == -1).sum().sum() / (results.shape[0] * results.shape[1]))
    # >> 0.6929338169796397

    visualize_response_matrix(results, results, "response_matrix")

    # save the results
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

data = torch.tensor(results.values, dtype=torch.float, device=device)
n_test_takers, n_items = data.shape
data_idtor = (data != -1).to(float)
data_ = data * data_idtor

valid_condition = False
trial = 0
while not valid_condition:
    train_idtor = torch.bernoulli(data_idtor * 0.8).int()
    test_idtor = data_idtor - train_idtor
    valid_condition = (train_idtor.sum(axis=1) != 0).all()
    valid_condition = valid_condition and (train_idtor.sum(axis=0) != 0).all()
    print(f"trial {trial} valid condition: {valid_condition}")
    trial += 1

embedding_name = "unique_prompts_embeddings_gte-Qwen2-7B-instruct.pkl"
with open(f"{file_path}/{embedding_name}", "rb") as f:
    prompt_embedding = pickle.load(f)

prompt_embedding.rename(columns={"question": "request.prompt"}, inplace=True)
merged_df = pd.merge(prompt_embedding, results_full, on="request.prompt", how="outer")
embedding = merged_df[["scenario", "request.prompt", "embedding"]]
# set the index to the request.prompt and scenario
embedding = embedding.set_index(["request.prompt", "scenario"])
# embedding = merged_df.groupby(["instance_id", "groups"])["embedding"]#.first()
embedding = embedding.loc[results.columns]

feature_type = "embedding"

if feature_type == "perplexity":
    # perplexity = results_full.groupby(["instance_id", "groups"])["perplexity"].min()
    results_full = results_full.set_index(["request.prompt", "scenario"])
    perplexity = results_full[["perplexity"]]
    perplexity = perplexity.loc[results.columns]

    features = torch.tensor(
        np.nan_to_num(perplexity, nan=0.0), dtype=torch.float, device=device
    )
    # >>> n_items
    features = features[:, None]
    # >>> n_items, 1

    has_feature = torch.tensor(~np.isnan(perplexity), dtype=torch.float, device=device)
else:
    has_feature = torch.tensor(
        ~embedding["embedding"].isna(), dtype=torch.float, device=device
    )

    # Convert embeddings to a tensor
    embedding_values = embedding["embedding"].apply(
        lambda x: x if isinstance(x, list) else [0.0] * len(embedding["embedding"][0])
    )
    features = torch.tensor(
        embedding_values.to_list(), dtype=torch.float, device=device
    )
    # >>>  n_items, embedding_dim

    # id = [i for i in range(has_feature.shape[0]) if has_feature[i] == 1]
    # features = torch.tensor(embedding["embedding"].to_list(), dtype=torch.float, device=device)

# has_feature is a binary vector, indicating if a the item has a feature or not
# Among test takers who have feature, I want to use a subset 80% of them to be the has_feature_train
# and 20% of them to be the has_feature_test
has_feature_train = torch.bernoulli(has_feature * 0.8).int()
has_feature_test = (has_feature - has_feature_train).int()
print("Fraction of data with perplexity feature: ", has_feature_train.float().mean())
# >>> 0.5345

w = torch.randn(features.shape[1], requires_grad=True, device=device)
b = torch.randn(1, requires_grad=True, device=device)
z_free = torch.zeros(n_items, requires_grad=True, device=device)
optim_z = LBFGS(
    [z_free, w, b], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
)
thetas_nuisance = torch.randn(150, n_test_takers, device=device)


def closure_z():
    optim_z.zero_grad()
    z = z_free * (1 - has_feature_train) + (features @ w + b) * has_feature_train
    probs = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
    loss = -(Bernoulli(probs=probs).log_prob(data_) * train_idtor).mean()
    loss.backward()
    return loss


z_free, w, b = trainer([z_free, w, b], optim_z, closure_z)
z = z_free * (1 - has_feature) + (features @ w + b) * has_feature
z = z.detach()
perplexity_rasch_z = z.clone()

thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
optim_theta = LBFGS(
    [thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
)


def closure_theta():
    optim_theta.zero_grad()
    probs = torch.sigmoid(thetas[:, None] + z[None, :])
    loss = -(Bernoulli(probs=probs).log_prob(data_) * train_idtor).mean()
    loss.backward()
    return loss


thetas = trainer([thetas], optim_theta, closure_theta)[0]
perplexity_rasch_thetas = thetas.clone()

# compute AUC ROC on train and test
probs = torch.sigmoid(thetas[:, None] + z[None, :])
compute_auc(probs, data_, train_idtor, test_idtor)
visualize_response_matrix(
    results, probs.detach().cpu().numpy(), f"response_matrix_prob_{feature_type}"
)

# trial 0 valid condition: False
# trial 1 valid condition: False
# trial 2 valid condition: True
# Fraction of data with perplexity feature:  tensor(0.8004, device='cuda:7')
# 100%|██████████| 100/100 [01:19<00:00,  1.25it/s, grad_norm=0.00133, d_params=0, d_loss=0]
#  17%|█▋        | 17/100 [00:00<00:02, 27.87it/s, grad_norm=1.54e-7, d_thetas=0, d_loss=1.05e-10]
# train auc: 0.6750435829162598
# test auc: 0.6736481189727783

# Fraction of data with perplexity feature:  tensor(0.8010, device='cuda:5')
# 100%|██████████| 100/100 [19:15<00:00, 11.55s/it, grad_norm=2.23e-5, d_parameter=1.25, d_loss=7.81e-7]
#  13%|█▎        | 13/100 [00:00<00:03, 24.07it/s, grad_norm=1.5e-7, d_parameter=0, d_loss=4.8e-10]
# train auc: 0.6958008408546448
# test auc: 0.69359290599823
