import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from utils import item_response_fn_1PL
from scipy.stats import gaussian_kde

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

    # Figure and axis setup for density plot on the right
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()  # Create a secondary y-axis on the right side for density plot

    theta_hats_all = []
    y_means = []
    
    for i in range(20):
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
        for i in range(4000):
            prob = item_response_fn_1PL(z_sub, theta_hat)
            loss = -torch.distributions.Bernoulli(
                probs=prob[sub_mask]
            ).log_prob(y_sub[sub_mask]).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            theta_hats.append(theta_hat.item())

        theta_hats_all.extend(theta_hats)
        y_means.append(y_sub[sub_mask].mean().item() * 6 - 3)

        # IRT line
        ax1.plot(theta_hats, color='red', alpha=0.5)
        # CTT dashed line
        ax1.hlines(y_sub[sub_mask].mean().item()*6-3, 0, 4000, color='blue', linestyle='dashed', alpha=0.5)

    # Final IRT and CTT lines
    ax1.hlines(theta, 0, 4000, color='red', linewidth=4)
    ax1.hlines(y[mask].mean().item()*6-3, 0, 4000, color='blue', linestyle='dashed', linewidth=4)

    ax1.set_ylim(-4, 4)

    # KDE for theta_hats
    kde_theta_hats = gaussian_kde(theta_hats_all)
    theta_grid = np.linspace(-4, 4, 100)
    ax2.plot(kde_theta_hats(theta_grid), theta_grid, color='red', alpha=0.8)

    # KDE for y_means
    kde_y_means = gaussian_kde(y_means)
    ax2.plot(kde_y_means(theta_grid), theta_grid, color='blue', linestyle='dashed', alpha=0.8)

    ax2.set_ylim(-4, 4)
    ax2.set_ylabel('Density')

    plt.savefig(f"{plot_dir}/hard_easy_test_{args.dataset}.png", dpi=300, bbox_inches='tight')
    plt.show()
