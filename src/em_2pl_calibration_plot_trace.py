import argparse
import os
import numpy as np
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from utils import item_response_fn_2PL, set_seed, goodness_of_fit_2PL_plot, theta_corr_ctt_plot
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def plot_corr(em_new_2pl_z2, true_2pl_z2, path):
    plt.figure(figsize=(10, 10))
    plt.scatter(em_new_2pl_z2, true_2pl_z2)
    plt.xlabel(r'$z2$ from EM 2PL calibration', fontsize=45)
    plt.ylabel(r'True $z2$', fontsize=45)
    plt.xlim(0,8)
    plt.ylim(0,8)
    corr, _ = spearmanr(true_2pl_z2, em_new_2pl_z2)
    plt.title(f'Correlation: {corr:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    
def em_2pl_calibration(
    response_matrix: torch.Tensor,
    true_2pl_z2: np.ndarray,
    true_2pl_z3: np.ndarray,
    max_epoch: int=3000,
    num_sample: int=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    num_model, num_item = response_matrix.shape
    
    # theta_nodes, weights = np.polynomial.hermite.hermgauss(num_node)
    # theta_nodes = torch.tensor(theta_nodes, device=device)
    # theta_matrix = theta_nodes[None, :].repeat(num_model, 1) # (num_model, num_node)
    # weights = torch.tensor(weights, device=device)
    
    theta_matrix = torch.normal(
        mean=0.0, std=1.0,
        size=(num_model, num_sample),
        requires_grad=True,
        device=device
    )
    
    z2_hat = torch.distributions.LogNormal(0.0, 0.5).sample(
        (num_item,)
    ).to(device).requires_grad_(True)
    z3_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(num_item,),
        requires_grad=True,
        device=device
    )
    
    # z2_hat = torch.tensor(
    #     true_2pl_z2,
    #     requires_grad=True,
    #     device=device
    # )
    # z3_hat = torch.tensor(
    #     true_2pl_z3,
    #     requires_grad=True,
    #     device=device
    # )
    optimizer = optim.Adam([z2_hat, z3_hat], lr=0.001)
    
    pbar = tqdm(range(max_epoch)) 
    for iter in pbar:
        theta_matrix_expand = theta_matrix.unsqueeze(1) # (num_model, 1, num_node)
        z2_hat_elu = F.elu(z2_hat, alpha=1.0)
        z2_hat_matrix = z2_hat_elu.unsqueeze(0)
        z3_hat_norm = (z3_hat - torch.mean(z3_hat)) / torch.std(z3_hat)
        z3_hat_matrix = z3_hat_norm.unsqueeze(0)
        
        # prob_matrixes = torch.zeros(num_model, num_item, num_sample)
        # for i in range(num_sample):
        #     prob_matrix = item_response_fn_2PL(z2_hat_matrix, z3_hat_matrix, theta_matrix_expand[:,:,i])
        #     prob_matrixes[:,:,i] = prob_matrix
            
        # agg_prob_matrix = torch.mean(prob_matrixes, dim=-1)
        
        prob_matrixes = item_response_fn_2PL(
            z2_hat_matrix.unsqueeze(2), # (1, num_item, 1)
            z3_hat_matrix.unsqueeze(2), # (1, num_item, 1)
            theta_matrix_expand, # (num_model, 1, num_sample)
        )
        agg_prob_matrix = torch.mean(prob_matrixes, dim=-1) 
        assert agg_prob_matrix.shape == response_matrix.shape
        
        # prob_matrixes = torch.zeros(num_model, num_item, num_node)
        # for i in range(num_node):
        #     prob_matrix = item_response_fn_2PL(z2_hat_matrix, z3_hat_matrix, theta_matrix_expand[:,:,i])
        #     mult = torch.exp(theta_nodes[i]**2 / 2) / torch.sqrt(torch.tensor(2 * torch.pi))
        #     prob_matrixes[:, :, i] = prob_matrix * mult * weights[i]
            
        # agg_prob_matrix = torch.sum(prob_matrixes, dim=-1)
        # assert agg_prob_matrix.shape == response_matrix.shape
        
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
        
        if iter % 10 == 0:
            plot_corr(z2_hat_elu.clone().detach().numpy(), true_2pl_z2, f"../plot/plot_trace/{iter}_z2.png")
            plot_corr(z3_hat_norm.clone().detach().numpy(), true_2pl_z3, f"../plot/plot_trace/{iter}_z3.png")

    return z2_hat_elu, z3_hat_norm

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
        theta_hat_norm = (theta_hat - torch.mean(theta_hat)) / torch.std(theta_hat)
        theta_hat_matrix = theta_hat_norm.unsqueeze(1)
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
        # wandb.log({'loss': loss.item()})
        
    return theta_hat_norm

if __name__ == "__main__":
    # wandb.init(project="em_2pl_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(1)
    input_dir = '../data/pre_calibration'
    output_dir = f'../data/em_2pl_calibration/{args.dataset}'
    plot_dir = f'../plot/em_2pl_calibration/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    y = pd.read_csv(f'{input_dir}/{args.dataset}/matrix.csv', index_col=0).values
    
    true_2pl_z2 = pd.read_csv('../data/pre_calibration/synthetic_2PL/z_true.csv')['z2'].values
    true_2pl_z3 = pd.read_csv('../data/pre_calibration/synthetic_2PL/z_true.csv')['z3'].values
    z2_hat, z3_hat = em_2pl_calibration(torch.tensor(y, dtype=torch.float64), true_2pl_z2, true_2pl_z3)
    
    z_df = pd.DataFrame({
        "z2": z2_hat.cpu().detach().numpy(),
        "z3": z3_hat.cpu().detach().numpy(),
    })
    z_df.to_csv(f"{output_dir}/z.csv", index=False)
    
    z2_hat = torch.tensor(pd.read_csv(f"{output_dir}/z.csv")['z2'].values)
    z3_hat = torch.tensor(pd.read_csv(f"{output_dir}/z.csv")['z3'].values)
    
    theta_hat = fit_theta_mle(
        response_matrix=torch.tensor(y, dtype=torch.float64),
        z2=z2_hat.clone().detach(),
        z3=z3_hat.clone().detach(),
    )
    theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=["theta"])
    theta_df.to_csv(f"{output_dir}/theta.csv", index=False)
    
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
    