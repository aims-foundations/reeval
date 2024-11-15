import argparse
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from utils.irt import IRT
from utils.utils import set_seed, inverse_sigmoid


def plot_hard_easy(
    theta_hats: list,
    y_means: list,
    theta: float,
    y_mean: float, 
    plot_path: str,
):
    plt.figure(figsize=(8, 6))
    plt.hist(theta_hats, bins=40, color='red', alpha=0.2, label='IRT Estimation', density=True)
    plt.hist(y_means, bins=40, color='blue', alpha=0.2, label='CTT Estimation', density=True)
    plt.axvline(x=theta, color='red', linestyle='-', linewidth=2)
    plt.axvline(x=y_mean, color='blue', linewidth=2)
    sns.kdeplot(theta_hats, color='red', linewidth=2, bw_adjust=2)
    plt.xlabel(r'Ability', fontsize=25)
    plt.xlim(-6,6)
    plt.ylabel(r'Density', fontsize=25)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"])
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    selection_prob = 0.8
    subset_size = 100
    step_size = 4000
    iterations = 1024
    batch_size = 1024
    assert batch_size % 2 == 0, "Batch size should divide by 2"
    
    y = pd.read_csv(f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0).values
    y = torch.tensor(y, device=device).float()
    
    abilities = pickle.load(open(f"../data/{args.fitting_method}_{args.PL}pl_calibration/{args.dataset}/abilities.pkl", "rb"))
    abilities = torch.tensor(abilities, device=device)
    
    item_parms = pickle.load(open(f"../data/{args.fitting_method}_{args.PL}pl_calibration/{args.dataset}/item_parms.pkl", "rb"))
    item_parms = torch.tensor(item_parms, device=device)
    z = item_parms[:, 0]
    
    assert y.shape[1] == z.shape[0], f"y.shape[1]: {y.shape[1]}, z.shape[0]: {z.shape[0]}"
    assert y.shape[0] == abilities.shape[0], f"y.shape[0]: {y.shape[0]}, theta.shape[0]: {abilities.shape[0]}"

    # rows in y with more than 500 non -1 values
    valid_rows_mask = (y != -1).sum(axis=1) > 500
    theta = abilities[valid_rows_mask]
    y = y[valid_rows_mask]
    
    # theta value closest to zero
    min_index = torch.argmin(torch.abs(theta))
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
        theta_hat = torch.normal(0, 1, size=(curr_batch_size, 1, args.D), requires_grad=True, device=device)
        optim = torch.optim.SGD([theta_hat], lr=0.01)
        
        for _ in tqdm(range(step_size)):
            prob = IRT.compute_prob(theta_hat, z_sub).squeeze(1)
            loss = -torch.distributions.Bernoulli(probs=prob).log_prob(y_sub).mean(-1).mean(0)
            optim.zero_grad()
            loss.backward()
            optim.step()

        theta_hats.extend(theta_hat.flatten().tolist())
        y_means.extend(inverse_sigmoid(y_sub.mean(-1)).tolist())
    
    save_dir = f'../data/hard_easy_test/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "theta_hat": theta_hats,
        "y_mean": y_means,
    })
    df.to_csv(f'{save_dir}/hard_easy_test.csv', index=False)
    
    theta_hats = pd.read_csv(f'../data/hard_easy_test/{args.dataset}/hard_easy_test.csv')["theta_hat"].values
    y_means = pd.read_csv(f'../data/hard_easy_test/{args.dataset}/hard_easy_test.csv')["y_mean"].values
    
    plot_dir = f'../plot/hard_easy_test'
    os.makedirs(plot_dir, exist_ok=True)
    plot_hard_easy(
        theta_hats,
        y_means,
        theta.cpu(), 
        inverse_sigmoid(y.mean()).item(), 
        f'{plot_dir}/hard_easy_{args.dataset}.png',
    )