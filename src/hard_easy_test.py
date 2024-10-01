import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from utils import item_response_fn_1PL
from matplotlib import gridspec

if __name__ == "__main__":
    wandb.init(project="hard_easy_test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    plot_dir = f'../plot/hard_easy_test'
    os.makedirs(plot_dir, exist_ok=True)
    
    selection_prob = 0.8
    subset_size = 100
    step_size = 10000
    test_size = 100
    
    response_matrix = pd.read_csv(
        f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0
    ).values
    count_minus_one = np.sum(response_matrix == -1, axis=1)
    min_index = np.argmin(count_minus_one)
    y = torch.tensor(response_matrix[min_index], dtype=torch.float32).to(device)
    mask = y != -1
    
    theta = pd.read_csv(
        f'../data/nonamor_calibration/{args.dataset}/nonamor_theta.csv'
    )["theta"].values[min_index]
    
    z = pd.read_csv(
        f'../data/nonamor_calibration/{args.dataset}/nonamor_z.csv'
    )["z"].values
    z = torch.tensor(z, dtype=torch.float32).to(device)
    z_sort_index = torch.argsort(z)

    theta_hats_all = []
    y_means_all = []
    
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 
    ax_main = plt.subplot(gs[0])
    
    for i in range(test_size):
        z_sort_index = torch.flip(z_sort_index, dims=[0])
        
        count = 0
        id = 0
        subset_index = []
        while count < subset_size:
            if torch.rand(1) < selection_prob:
                subset_index.append(z_sort_index[id].item())
                count = count + 1
            id = id + 1
            
        z_sub = z[subset_index]
        y_sub = y[subset_index]
        sub_mask = y_sub != -1

        theta_hat = torch.normal(
            0, 1, size=(1,), requires_grad=True, device=device
        )
        optim = torch.optim.SGD([theta_hat], lr=0.01)
        
        losses = []
        theta_hats = []
        for i in range(step_size):
            prob = item_response_fn_1PL(z_sub, theta_hat)
            loss = -torch.distributions.Bernoulli(
                probs=prob[sub_mask]
            ).log_prob(y_sub[sub_mask]).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            theta_hats.append(theta_hat.item())
        
        ax_main.plot(theta_hats, color='red', alpha=0.3)
        ax_main.hlines(y_sub[sub_mask].mean().item()*6-3, 0, step_size, color='blue', linestyle='dashed', alpha=0.3)
        theta_hats_all.append(theta_hat.item())
        y_means_all.append(y_sub[sub_mask].mean().item()*6-3)
   
    ax_main.hlines(theta, 0, step_size, color='red', linewidth=4)
    ax_main.hlines(y[mask].mean().item()*6-3, 0, step_size, color='blue', linestyle='dashed', linewidth=4)
    ax_main.set_ylim(-4, 4)
    ax_main.set_xlim(0, step_size)
    ax_main.set_xlabel('Step', fontsize=25)
    ax_main.set_ylabel('Ability', fontsize=25)
    ax_main.tick_params(axis='both', labelsize=25)
    
    ax_hist_theta = plt.subplot(gs[1], sharey=ax_main)
    ax_hist_theta.hist(theta_hats_all, bins=30, color='red', alpha=0.3, orientation='horizontal')
    ax_hist_theta.axis('off')
    
    ax_hist_theta.hist(y_means_all, bins=30, color='blue', alpha=0.3, orientation='horizontal')
    plt.savefig(f"{plot_dir}/hard_easy_test_{args.dataset}.png", dpi=300, bbox_inches='tight')
