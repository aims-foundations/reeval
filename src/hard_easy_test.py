import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from utils import item_response_fn_1PL

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
    step_size = 1000
    test_size = 5
    
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
        
        theta_hats_all.append(theta_hat.item())
        y_means_all.append(y_sub[sub_mask].mean().item() * 6 - 3)
    
    # save theta_hats_all and y_means_all in csv
    df = pd.DataFrame({'theta_hat': theta_hats_all, 'y_mean': y_means_all})
    df.to_csv(f'../data/hard_easy_test/{args.dataset}.csv', index=False)
    
    plt.figure(figsize=(8, 6))
    plt.hist(theta_hats_all, bins=30, color='red', alpha=0.5, label='Theta Distribution', density=True)
    plt.hist(y_means_all, bins=30, color='blue', alpha=0.5, label='CTT Distribution', density=True)
    sns.kdeplot(theta_hats_all, color='red', label='Theta KDE', linewidth=2)
    sns.kdeplot(y_means_all, color='blue', label='CTT KDE', linewidth=2)
    plt.axvline(x=theta, color='red', linestyle='-', linewidth=2, label='True Theta')
    plt.axvline(x=y[mask].mean().item() * 6 - 3, color='blue', linestyle='--', linewidth=2, label='True CTT Score')
    plt.xlabel('Ability / CTT Score', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.title('Theta and CTT Score Distributions with KDE', fontsize=18)
    plt.legend(fontsize=12)
    plt.savefig(f"{plot_dir}/hard_easy_test_{args.dataset}.png", dpi=300, bbox_inches='tight')
    plt.show()
