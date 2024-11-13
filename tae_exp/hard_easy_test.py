import argparse
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import item_response_fn_1PL, set_seed, inverse_sigmoid, plot_hard_easy

if __name__ == "__main__":
    wandb.init(project="hard_easy_test_new_em")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--calibration_method', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    selection_prob = 0.8
    subset_size = 100
    step_size = 4000
    iterations = 1024
    batch_size = 1024
    assert batch_size % 2 == 0, "Batch size should divide by 2"
    
    y = pd.read_csv(f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0).values

    if 
    theta = pd.read_csv(f'../data/em_1pl_calibration/{args.dataset}/theta.csv')["theta"].values
    z = pd.read_csv(f'../data/em_1pl_calibration/{args.dataset}/z.csv')["z"].values
    assert y.shape[1] == z.shape[0], f"y.shape[1]: {y.shape[1]}, z.shape[0]: {z.shape[0]}"
    assert y.shape[0] == theta.shape[0], f"y.shape[0]: {y.shape[0]}, theta.shape[0]: {theta.shape[0]}"

    # rows in y with more than 500 non -1 values
    valid_rows_mask = (y != -1).sum(axis=1) > 500
    theta = theta[valid_rows_mask]
    y = y[valid_rows_mask]
    
    # theta value closest to zero
    min_index = np.argmin(np.abs(theta))
    y = y[min_index]
    theta = theta[min_index]
    assert y.shape == z.shape, f"y.shape: {y.shape}, z.shape: {z.shape}"
    
    valid_cols_mask = y != -1
    y = y[valid_cols_mask]
    z = z[valid_cols_mask]
    assert y.shape == z.shape, f"y.shape: {y.shape}, z.shape: {z.shape}"
    
    y = torch.tensor(y, dtype=torch.float64).to(device)
    z = torch.tensor(z, dtype=torch.float64).to(device)
    
    z_sort_index = torch.argsort(z)
    z_sorted = z[z_sort_index]
    
    mean_z_sorted = z_sorted[0]
    std_z_sorted = abs(z_sorted[-1] - z_sorted[0]) / 10

    idx_prob = (1 / (std_z_sorted * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-((z_sorted - mean_z_sorted) ** 2) / (2 * std_z_sorted ** 2))
    idx_prob = torch.clamp(idx_prob, min=1 / (z.shape[0] * 5))
    idx_prob = idx_prob / idx_prob.sum()
    
    theta_hats = []
    y_means = []
    for batch_start in range(0, iterations, batch_size):
        batch_end = min(batch_start + batch_size, iterations)
        print(f"Processing iteration {batch_start}-{batch_end}")
        curr_batch_size = batch_end - batch_start
        
        expanded_probs = idx_prob[None].repeat(curr_batch_size, 1)
        chosen_flip_dims = torch.randint(low=0, high=curr_batch_size, size=torch.Size([batch_size//2]))
        expanded_probs[chosen_flip_dims] = torch.flip(expanded_probs[chosen_flip_dims], dims=[-1])
        
        subset_index = torch.multinomial(expanded_probs, num_samples=100, replacement=False)
        z_sub = z_sorted[subset_index]
        y_sub = y[z_sort_index[subset_index]]
        
        # Batch processing
        theta_hat = torch.normal(0, 1, size=(curr_batch_size,1), requires_grad=True, device=device)
        optim = torch.optim.SGD([theta_hat], lr=0.01)
        
        for _ in tqdm(range(step_size)):
            prob = item_response_fn_1PL(z_sub, theta_hat)
            loss = -torch.distributions.Bernoulli(probs=prob).log_prob(y_sub).mean(-1).mean(0)
            optim.zero_grad()
            loss.backward()
            optim.step()

        wandb.log({'loss': loss.item()})
        theta_hats.extend(theta_hat.flatten().tolist())
        y_means.extend(inverse_sigmoid(y_sub.mean(-1)).tolist())
    
    save_dir = f'../data/hard_easy_test_new_em/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "theta_hat": theta_hats,
        "y_mean": y_means,
    })
    df.to_csv(f'{save_dir}/hard_easy_test_new_em.csv', index=False)
    
    theta_hats = pd.read_csv(f'../data/hard_easy_test_new_em/{args.dataset}/hard_easy_test_new_em.csv')["theta_hat"].values
    y_means = pd.read_csv(f'../data/hard_easy_test_new_em/{args.dataset}/hard_easy_test_new_em.csv')["y_mean"].values
    
    plot_dir = f'../plot/hard_easy_test_new_em'
    os.makedirs(plot_dir, exist_ok=True)
    plot_hard_easy(
        theta_hats,
        y_means,
        theta, 
        inverse_sigmoid(y.mean()).item(), 
        f'{plot_dir}/hard_easy_{args.dataset}.png',
    )