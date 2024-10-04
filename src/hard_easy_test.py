import argparse
import os
import numpy as np
import torch
import pandas as pd
import wandb
from utils import item_response_fn_1PL, set_seed

if __name__ == "__main__":
    wandb.init(project="hard_easy_test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    selection_prob = 0.8
    subset_size = 100
    step_size = 4000
    iterations = 100
    
    response_matrix = pd.read_csv(
        f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0
    ).values
    theta = pd.read_csv(
        f'../data/nonamor_calibration/{args.dataset}/nonamor_theta.csv'
    )["theta"].values
    z = pd.read_csv(
        f'../data/nonamor_calibration/{args.dataset}/nonamor_z.csv'
    )["z"].values

    # rows in y with more than 500 non -1 values
    valid_rows_mask = (response_matrix != -1).sum(axis=1) > 500
    filtered_theta = theta[valid_rows_mask]
    filtered_response_matrix = response_matrix[valid_rows_mask]
    # theta value closest to zero
    min_index = np.argmin(np.abs(filtered_theta))
    
    y = torch.tensor(filtered_response_matrix[min_index], dtype=torch.float32).to(device)
    valid_cols_mask = y != -1
    y = y[valid_cols_mask]
    z = torch.tensor(z[valid_cols_mask], dtype=torch.float32).to(device)
    theta = torch.tensor(filtered_theta[min_index], dtype=torch.float32).to(device)
    
    z_sort_index = torch.argsort(z)

    theta_hats_all = []
    y_means_all = []
    for iteration in range(iterations):
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
        for step in range(step_size):
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
        wandb.log({
            'iteration': iteration,
            'step': step,
            'loss': loss.item(),
        })
    
    save_dir = f'../data/hard_easy_test/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame({
        "theta_hat": theta_hats_all,
        "y_mean": y_means_all
    })
    df.to_csv(f'{save_dir}/hard_easy_test.csv', index=False)