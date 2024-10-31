import argparse
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import wandb
from utils import item_response_fn_1PL, set_seed, inverse_sigmoid, plot_hard_easy

if __name__ == "__main__":
    wandb.init(project="hard_easy_test_old_em")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    selection_prob = 0.8
    subset_size = 100
    step_size = 4000
    iterations = 1000
    
    y = pd.read_csv(f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0).values
    theta = pd.read_csv(f'../data/em_1pl_calibration/{args.dataset}/theta.csv')["theta"].values
    z = pd.read_csv(f'../data/em_1pl_calibration/{args.dataset}/z.csv')["z"].values
    assert y.shape[1] == z.shape[0], f"y.shape[1]: {y.shape[1]}, z.shape[0]: {z.shape[0]}"
    assert y.shape[0] == theta.shape[0], f"y.shape[0]: {y.shape[0]}, theta.shape[0]: {theta.shape[0]}"
    
    count_minus_one = np.sum(y == -1, axis=1)
    min_index = np.argmin(count_minus_one)
    y = y[min_index]
    theta = theta[min_index]
    assert y.shape == z.shape, f"y.shape: {y.shape}, z.shape: {z.shape}"
    
    valid_cols_mask = y != -1
    y = y[valid_cols_mask]
    z = z[valid_cols_mask]
    assert y.shape == z.shape, f"y.shape: {y.shape}, z.shape: {z.shape}"
    
    y = torch.tensor(y, dtype=torch.float32).to(device)
    z = torch.tensor(z, dtype=torch.float32).to(device)
    
    z_sort_index = torch.argsort(z)

    theta_hats = []
    y_means = []
    for _ in tqdm(range(iterations)):
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

        theta_hat = torch.normal(0, 1, size=(1,), requires_grad=True, device=device)
        optim = torch.optim.SGD([theta_hat], lr=0.01)
        
        for _ in range(step_size):
            prob = item_response_fn_1PL(z_sub, theta_hat)
            loss = -torch.distributions.Bernoulli(probs=prob).log_prob(y_sub).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            
        wandb.log({'loss': loss.item()})
        theta_hats.append(theta_hat.item())
        y_means.append(inverse_sigmoid(y_sub.mean()).item())
    
    save_dir = f'../data/hard_easy_test_old_em/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "theta_hat": theta_hats,
        "y_mean": y_means,
    })
    df.to_csv(f'{save_dir}/hard_easy_test_old_em.csv', index=False)
    
    plot_dir = f'../plot/hard_easy_test_old_em'
    os.makedirs(plot_dir, exist_ok=True)
    plot_hard_easy(
        theta_hats,
        y_means,
        theta, 
        inverse_sigmoid(y.mean()).item(), 
        f'{plot_dir}/hard_easy_old_{args.dataset}.png',
    )