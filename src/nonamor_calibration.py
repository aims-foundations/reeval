import argparse
import os
import torch
import wandb
from utils import set_seed
import pandas as pd
from tqdm import tqdm
from utils import item_response_fn_1PL
import torch.optim as optim

def nonamor_calibration(
    response_matrix: torch.Tensor,
    max_epoch: int=3000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    theta_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(0),),
        requires_grad=True,
        device=device
    )
    z_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(1),),
        requires_grad=True,
        device=device
    )

    optimizer = optim.Adam([theta_hat, z_hat], lr=0.01)
    
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        theta_hat_matrix = theta_hat.unsqueeze(1)
        z_hat_matrix = z_hat.unsqueeze(0)
        prob_matrix = item_response_fn_1PL(z_hat_matrix, theta_hat_matrix)
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item()})

    return theta_hat, z_hat

if __name__ == "__main__":
    wandb.init(project="nonamor_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    input_dir = '../data/pre_calibration/'
    output_dir = f'../data/nonamor_calibration/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    y = pd.read_csv(f'{input_dir}/{args.dataset}/matrix.csv', index_col=0).values
    theta_hat, z_hat = nonamor_calibration(torch.tensor(y, dtype=torch.float32))
    
    z_df = pd.DataFrame(z_hat.cpu().detach().numpy(), columns=["z"])
    z_df.to_csv(f"{output_dir}/nonamor_z.csv", index=False)
    theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=["theta"])
    theta_df.to_csv(f"{output_dir}/nonamor_theta.csv", index=False)
    