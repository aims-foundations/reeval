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


from torchmetrics import AUROC
auroc = AUROC(task="binary")

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)

device = "cuda:0"

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

def load_old_benchmark():
    with open(f"/lfs/skampere1/0/sttruong/reeval/data/resmat.pkl", "rb") as f:
        results = pickle.load(f)
        
    # data_withnan, missing=nan
    # data_withneg1, missing=-1
    # data_with0, missing=0
    data_withnan = torch.tensor(results.values, dtype=torch.float, device=device)
    data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
    data_idtor = (data_withneg1 != -1).to(float)
    data_with0 = data_withneg1 * data_idtor # -1 -> 0
    valid_condition = False
    trial = 0
    while not valid_condition:
        train_idtor = torch.bernoulli(data_idtor * 0.8).int()
        test_idtor = data_idtor - train_idtor
        valid_condition = (train_idtor.sum(axis=1) != 0).all() and (train_idtor.sum(axis=0) != 0).all()
        print(f"trial {trial} valid condition: {valid_condition}")
        trial += 1
    

    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool()


def get_new_benchmark():
    all_benchmark_data = pd.read_csv("/lfs/skampere1/0/sttruong/reeval/calibration/all_benchmarks_joined.csv")
    
    data_clean = all_benchmark_data.drop('model_name', axis=1)

    # Find binary columns
    binary_cols = []
    for col in data_clean.columns:
        unique_vals = data_clean[col].dropna().unique()
        if all(val in [0.0, 1.0] for val in unique_vals):
            binary_cols.append(col)
    # Keep only binary columns
    data_binary = data_clean[binary_cols]

    # Convert ALL columns to float64 (including boolean columns)
    data_binary = data_binary.astype('float64')

    # Now convert to tensor
    data_withnan = torch.tensor(data_binary.values, dtype=torch.float32)
    
    data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
    data_idtor = (data_withneg1 != -1).to(float)
    data_with0 = data_withneg1 * data_idtor # -1 -> 0
    trial = 0
    valid_condition = False
    while not valid_condition:
        train_idtor = torch.bernoulli(data_idtor * 0.8).int()
        test_idtor = data_idtor - train_idtor
        valid_condition = (train_idtor.sum(axis=1) != 0).all() and (train_idtor.sum(axis=0) != 0).all()
        print(f"trial {trial} valid condition: {valid_condition}")
        trial += 1
    
    
    
    
    return data_withneg1, data_with0, data_idtor.bool(), train_idtor.bool(), test_idtor.bool()
