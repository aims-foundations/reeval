import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from utils import item_response_fn_1PL

if __name__ == "__main__":
    wandb.init(project="hard_easy_test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    plot_dir = f'../plot/hard_easy_test'
    os.makedirs(plot_dir, exist_ok=True)
    
    selection_prob = 0.8
    subset_size = 100
    
    response_matrix = pd.read_csv(f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0).values
    count_minus_one = np.sum(response_matrix == -1, axis=1)
    min_index = np.argmin(count_minus_one)
    y = response_matrix[min_index]
    y = torch.tensor(y, dtype=torch.float32)

    theta = pd.read_csv(
        f'../data/nonamor_calibration/{args.dataset}/nonamor_theta.csv'
    )["theta"].values[min_index]
    
    z = pd.read_csv(f'../data/nonamor_calibration/{args.dataset}/nonamor_z.csv')["z"].values
    z = torch.tensor(z, dtype=torch.float32)
    z_sort_index = torch.argsort(z)

    for i in range(20):
        z_sort_index = torch.flip(z_sort_index, dims=[0])
        
        count = 0
        id = 0
        subset_index = []
        while count < subset_size:
            if torch.rand(1) < selection_prob:
                subset_index.append(z_sort_index[id])
                count = count + 1
            id = id + 1
            
        z_sub = z[subset_index]
        y_sub = y[subset_index]

        theta_hat = torch.normal(0, 1, size=(1,), requires_grad=True)
        optim = torch.optim.SGD([theta_hat], lr=0.01)
        losses = []
        theta_hats = []
        for i in range(4000):
            mask = y_sub != -1
            prob = item_response_fn_1PL(z_sub, theta_hat)
            loss = -torch.distributions.Bernoulli(
                probs=prob[mask]
            ).log_prob(y_sub[mask]).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            theta_hats.append(theta_hat.item())

        #irt
        plt.plot(theta_hats, color='red', alpha=0.5)
        #ctt
        plt.hlines(y_sub.mean().item()*6-3, 0, 4000, color='blue', linestyle='dashed', alpha=0.5)

    #irt
    plt.hlines(theta, 0, 4000, color='red', linewidth=4)
    #ctt
    plt.hlines(y.mean().item()*6-3, 0, 4000, color='blue', linestyle='dashed', linewidth=4)

    plt.ylim(-4, 4)
    plt.savefig(f"{plot_dir}/hard_easy_test_{args.dataset}.png", dpi=300, bbox_inches='tight')