import pandas as pd
import numpy as np
import torch
import pickle
import os
import json
import gc
from torch.distributions import Bernoulli
from torch.optim import LBFGS
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
bundles.icml2024()

from torchmetrics import AUROC
auroc = AUROC(task="binary")

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)

device = "cuda:2"

def visualize_response_matrix(results, value, filename):
    # Extract the groups labels in the order of the columns
    group_values = results.columns.get_level_values("scenario")

    # Identify the boundaries where the group changes
    boundaries = []
    for i in range(1, len(group_values)):
        if group_values[i] != group_values[i - 1]:
            boundaries.append(i - 0.5)  # using 0.5 to place the line between columns

    # Visualize the results with a matrix: red is 0, white is -1 and blue is 1
    cmap = mcolors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
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

    # Define the minimum spacing between labels (e.g., 100 units)
    min_spacing = 100
    last_label_pos = -float("inf")
    # Plot the matrix
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.matshow(value, aspect="auto", cmap=cmap, norm=norm)

        # Add vertical lines at each boundary
        for b in boundaries:
            ax.axvline(x=b, color="black", linewidth=0.25, linestyle="--", alpha=0.5)
        
        # Add group labels above the matrix, only if they're spaced enough apart
        for name, pos in zip(group_names, group_midpoints):
            if pos - last_label_pos >= min_spacing:
                ax.text(pos, -5, name, ha='center', va='bottom', rotation=90, fontsize=3)
                last_label_pos = pos

        # Add model labels on the y-axis
        ax.set_yticks(range(len(results.index)))
        ax.set_yticklabels(results.index, fontsize=3)

        # Add a colorbar
        cbar = plt.colorbar(cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["-1", "0", "1"])
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()

def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
    return parameters

def compute_auc(probs, data, train_idtor, test_idtor):
    train_probs = probs[train_idtor.bool()]
    test_probs = probs[test_idtor.bool()]
    train_labels = data[train_idtor.bool()]
    test_labels = data[test_idtor.bool()]
    
    train_auc = auroc(train_probs, train_labels)
    test_auc = auroc(test_probs, test_labels)
    print(f"train auc: {train_auc}")
    print(f"test auc: {test_auc}")
    
    return train_auc, test_auc

def compute_cttcorr(probs, data, train_idtor, test_idtor):
    train_probs  = probs.clone()
    test_probs   = probs.clone()
    train_labels = data.clone()
    test_labels  = data.clone()

    train_mask = ~train_idtor.bool()
    train_probs[train_mask]  = float('nan')
    train_labels[train_mask] = float('nan')

    test_mask = ~test_idtor.bool()
    test_probs[test_mask]   = float('nan')
    test_labels[test_mask]  = float('nan')
    
    train_prob_ctt = torch.nanmean(train_probs, dim=1).detach().cpu().numpy()
    train_label_ctt = torch.nanmean(train_labels, dim=1).detach().cpu().numpy()
    train_mask = ~np.isnan(train_prob_ctt) & ~np.isnan(train_label_ctt)
    train_cttcorr = pearsonr(train_prob_ctt[train_mask], train_label_ctt[train_mask]).statistic
    
    test_prob_ctt = torch.nanmean(test_probs, dim=1).detach().cpu().numpy()
    test_label_ctt = torch.nanmean(test_labels, dim=1).detach().cpu().numpy()
    test_mask = ~np.isnan(test_prob_ctt) & ~np.isnan(test_label_ctt)
    test_cttcorr = pearsonr(test_prob_ctt[test_mask], test_label_ctt[test_mask]).statistic
    
    print(f"train cttcorr: {train_cttcorr}")
    print(f"test cttcorr: {test_cttcorr}")

    return train_cttcorr, test_cttcorr


with open(f"/lfs/skampere1/0/sttruong/reeval/data/resmat.pkl", "rb") as f:
    results = pickle.load(f)
    
# data_withnan, missing=nan
# data_withneg1, missing=-1
# data_with0, missing=0
data_withnan = torch.tensor(results.values, dtype=torch.float, device=device)

data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
data_idtor = (data_withneg1 != -1).to(float)
data_with0 = data_withneg1 * data_idtor # -1 -> 0
n_test_takers, n_items = data_with0.shape
scenarios = results.columns.get_level_values("scenario").unique()

# save dict
metric_results = defaultdict(dict)


# load this file: resmat.pt
resmat = torch.load("/lfs/skampere1/0/sttruong/reeval/resmat.pt")
P = resmat["data_tensor"].mean(-1)
P1 = P[0]


# sampling without replacement for P1 with sample size of 50. Repeat 1000 times
# for each sample, compute the mean
# plot a histogram of the means

sample_size = 80
n_samples = 10000
means = []
for _ in range(n_samples):
    sample = P1[torch.randperm(P1.shape[0])[:sample_size]]
    means.append(sample.mean().item())
plt.hist(means, bins=50, density=True)
plt.xlabel("Mean of Sample")
plt.ylabel("Density")
plt.title(f"Histogram of Sample Means (n={sample_size}, N={n_samples})")
# x axis limits to 0 1
plt.xlim(0, 1)

# Plot the mean of P1
plt.axvline(P1.mean().item(), color='red', linestyle='dashed', linewidth=1, label='Mean of P1')
plt.legend()
plt.show()    

vis_resmat_dir = "../result/visualize_resmat"
os.makedirs(vis_resmat_dir, exist_ok=True)

# overall stats
print("Number of test takers:", results.shape[0])
print("Number of items:", results.shape[1])
print("Number of scenarios:", results.columns.get_level_values("scenario").nunique())
visualize_response_matrix(results, results, f"{vis_resmat_dir}/resmat_all")

# count the number of items and test takers in each dataset
scenario_counts = {}
for scenario in sorted(scenarios):
    mask = results.columns.get_level_values("scenario") == scenario
    sub_results = results.loc[:, mask]
    scenario_counts[scenario] = {
        "n_items": sub_results.shape[1],
        "n_test_takers": sub_results.notna().any(axis=1).sum()
    }
    print(f"{scenario}: {scenario_counts[scenario]['n_test_takers']} test takers, {scenario_counts[scenario]['n_items']} items")
    # visualize_response_matrix(sub_results, sub_results, f"{vis_resmat_dir}/resmat_{scenario}")
    
# data_idtor = train_idtor + test_idtor
# apply random train/test mask to the matrix, and ensure no one row or column is fully masked
valid_condition = False
trial = 0
while not valid_condition:
    train_idtor = torch.bernoulli(data_idtor * 0.8).int()
    test_idtor = data_idtor - train_idtor
    valid_condition = (train_idtor.sum(axis=1) != 0).all() and (train_idtor.sum(axis=0) != 0).all()
    print(f"trial {trial} valid condition: {valid_condition}")
    trial += 1

# fit z
B = 50000
optimized_zs = []
thetas_nuisance = torch.randn(150, n_test_takers, device=device)

for i in tqdm(range(0, n_items, B)):
    data_batch = data_with0[:, i:i+B]
    train_idtor_batch = train_idtor[:, i:i+B]
    current_B = data_batch.shape[1]
    z_i = torch.randn(current_B, requires_grad=True, device=device)
    optim_z_i = LBFGS([z_i], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_z_i():
        optim_z_i.zero_grad()
        probs = torch.sigmoid(thetas_nuisance[:, :, None] + z_i[None, None, :])
        loss = -(Bernoulli(probs=probs).log_prob(data_batch)*train_idtor_batch).mean()
        loss.backward()
        return loss
    z_i_optimized = trainer([z_i], optim_z_i, closure_z_i)[0].detach()
    optimized_zs.append(z_i_optimized)
zs = torch.cat(optimized_zs)

# fit theta
thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
def closure_theta():
    optim_theta.zero_grad()
    probs = torch.sigmoid(thetas[:, None] + zs[None, :])
    loss = -(Bernoulli(probs=probs).log_prob(data_with0)*train_idtor).mean()
    loss.backward()
    return loss
thetas = trainer([thetas], optim_theta, closure_theta)[0]

# calculate metrics
probs = torch.sigmoid(thetas[:, None] + zs[None, :])

train_auc, test_auc = compute_auc(probs, data_with0, train_idtor, test_idtor)
metric_results["combined_data"]["train_auc"] = train_auc.item()
metric_results["combined_data"]["test_auc"] = test_auc.item()

train_cttcorr, test_cttcorr = compute_cttcorr(probs, data_with0, train_idtor, test_idtor)
metric_results["combined_data"]["train_cttcorr"] = train_cttcorr.item()
metric_results["combined_data"]["test_cttcorr"] = test_cttcorr.item()

del optim_theta, thetas, z_i, thetas_nuisance, optim_z_i
gc.collect()
torch.cuda.empty_cache()