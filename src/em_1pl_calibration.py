import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wandb
from utils import item_response_fn_1PL, set_seed
import torch.optim as optim

def em_calibration(
    response_matrix, 
    max_iter=100, 
    tolerance=1e-4, 
    num_node=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    num_model, num_item = response_matrix.shape
    
    nodes, weights = np.polynomial.hermite.hermgauss(num_node)
    nodes = torch.tensor(nodes, device=device)
    weights = torch.tensor(weights, device=device)
    
    z = torch.normal(
        mean=0.0, std=1.0,
        size=(num_item,),
        requires_grad=True,
        device=device
    )
    optimizer = optim.Adam([z,], lr=0.01)
        
    pbar = tqdm(range(max_iter))
    for iter in pbar:
        old_z = z.detach().clone()
        
        # E-step
        margin_prob = torch.zeros(num_item)
        for j in range(num_item):
            integrand = torch.zeros(num_node)
            for k, theta_node in enumerate(nodes):
                prob = item_response_fn_1PL(z[j], theta_node)
                # P(theta_i) = 1/sqrt(pi) * exp(-theta_i^2)
                p_theta_node = torch.exp(-theta_node**2) / torch.sqrt(torch.tensor(torch.pi))
                integrand[k] = prob * p_theta_node
        margin_prob[j] = torch.sum(weights * integrand) # (num_item,)
        margin_prob_matrix= margin_prob[None, :].repeat(num_model, 1) # (num_model, num_item)
        
        # M-step
        optimizer.zero_grad()
        bern_matrix = torch.distributions.Bernoulli(margin_prob_matrix)
        loss = -bern_matrix.log_prob(response_matrix).mean()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
        
        if torch.max(torch.abs(z - old_z)) < tolerance:
            break
    
    return z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    input_dir = '../data/pre_calibration/'
    output_dir = f'../data/em_1pl_calibration/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    y = pd.read_csv(f'{input_dir}/{args.dataset}/matrix.csv', index_col=0).values
    z_hat = em_calibration(torch.tensor(y, dtype=torch.float32))
    
    z_df = pd.DataFrame({
        'z': z_hat.cpu().detach().numpy(),
    })
    z_df.to_csv(f"{output_dir}/z.csv", index=False)
    
