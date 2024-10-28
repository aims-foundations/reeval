import argparse
import os
import numpy as np
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from utils import item_response_fn_2PL, set_seed, goodness_of_fit_2PL_plot, theta_corr_ctt_plot
import torch.optim as optim

def simu_em_2pl_calibration(
    response_matrix: torch.Tensor,
    max_epoch: int=3000,
    num_node: int=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    num_model, num_item = response_matrix.shape
    
    theta_nodes, weights = np.polynomial.hermite.hermgauss(num_node)
    theta_nodes = torch.tensor(theta_nodes, device=device)
    theta_matrix = theta_nodes[None, :].repeat(num_model, 1) # (num_model, num_node)
    weights = torch.tensor(weights, device=device)
    
    z2_hat = torch.distributions.LogNormal(0.0, 0.5).sample(
        (num_item,)
    ).to(device).requires_grad_(True)
    z3_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(num_item,),
        requires_grad=True,
        device=device
    )
    optimizer = optim.Adam([z2_hat, z3_hat], lr=0.01)
    
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        theta_matrix_expand = theta_matrix.unsqueeze(1) # (num_model, 1, num_node)
        z2_hat_matrix = z2_hat.unsqueeze(0)
        z3_hat_matrix = z3_hat.unsqueeze(0)
        prob_matrixes = torch.zeros(num_model, num_item, num_node)
        for i in range(num_node):
            prob_matrix = item_response_fn_2PL(z2_hat_matrix, z3_hat_matrix, theta_matrix_expand[:,:,i])
            mult = torch.exp(theta_nodes[i]**2 / 2) / torch.sqrt(torch.tensor(2 * torch.pi))
            prob_matrixes[:, :, i] = prob_matrix * mult * weights[i]
            
        agg_prob_matrix = torch.sum(prob_matrixes, dim=-1)
        assert agg_prob_matrix.shape == response_matrix.shape
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = agg_prob_matrix.flatten()[mask.flatten()]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item()})
        # wandb.log({'loss': loss.item()})

    return z2_hat, z3_hat

def fit_theta_mle(
    response_matrix: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
    max_epoch: int=3000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    num_model, num_item = response_matrix.shape
    
    theta_hat = torch.normal(
        mean=0.0, std=1.0, 
        size=(num_model,), 
        requires_grad=True, 
        device=device,
    )
    optimizer = optim.Adam([theta_hat], lr=0.01)
    
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        theta_hat_matrix = theta_hat.unsqueeze(1)
        z2_matrix = z2.unsqueeze(0)
        z3_matrix = z3.unsqueeze(0)
        prob_matrix = item_response_fn_2PL(z2_matrix, z3_matrix, theta_hat_matrix)
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix({'loss': loss.item()})
        
    return theta_hat

if __name__ == "__main__":
    # wandb.init(project="simu_em_2pl_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    input_dir = '../data/pre_calibration'
    output_dir = f'../data/simu_em_2pl_calibration/{args.dataset}'
    plot_dir = f'../plot/simu_em_2pl_calibration/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    y = pd.read_csv(f'{input_dir}/{args.dataset}/matrix.csv', index_col=0).values
    # z2_hat, z3_hat = simu_em_2pl_calibration(torch.tensor(y, dtype=torch.float64))
    
    # z_df = pd.DataFrame({
    #     "z2": z2_hat.cpu().detach().numpy(),
    #     "z3": z3_hat.cpu().detach().numpy(),
    # })
    # z_df.to_csv(f"{output_dir}/z.csv", index=False)
    
    # z2_hat = torch.tensor(pd.read_csv(f"{output_dir}/z.csv")['z2'].values)
    # z3_hat = torch.tensor(pd.read_csv(f"{output_dir}/z.csv")['z3'].values)
    
    # theta_hat = fit_theta_mle(
    #     response_matrix=torch.tensor(y, dtype=torch.float64),
    #     z2=z2_hat,
    #     z3=z3_hat,
    # )
    # theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=["theta"])
    # theta_df.to_csv(f"{output_dir}/theta.csv", index=False)
    
    z2_hat = torch.tensor(pd.read_csv(f"{output_dir}/z.csv")['z2'].values)
    z3_hat = torch.tensor(pd.read_csv(f"{output_dir}/z.csv")['z3'].values)
    theta_hat = torch.tensor(pd.read_csv(f"{output_dir}/theta.csv")['theta'].values)
    
    _, _ = goodness_of_fit_2PL_plot(
        z2=z2_hat.cpu().clone().detach(),
        z3=z3_hat.cpu().clone().detach(),
        theta=theta_hat.cpu().clone().detach(),
        y=torch.tensor(y, dtype=torch.float64),
        plot_path=f"{plot_dir}/goodness_of_fit_{args.dataset}.png"
    )
    
    _, _ = theta_corr_ctt_plot(
        theta=theta_hat.cpu().detach().numpy(),
        y=y,
        plot_path=f"{plot_dir}/theta_corr_ctt_{args.dataset}",
    )
    