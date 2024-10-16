import argparse
import os
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from utils import item_response_fn_3PL, set_seed
import torch.optim as optim

def mle_3pl_calibration(
    response_matrix: torch.Tensor,
    max_epoch: int=3000,
    patience: int=50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    theta_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(0),),
        requires_grad=True,
        device=device
    )
    z1_hat = torch.distributions.Beta(0.5, 4).sample(
        (response_matrix.size(1),)
    ).to(device).requires_grad_(True)
    z2_hat = torch.distributions.LogNormal(0.0, 1.0).sample(
        (response_matrix.size(1),)
    ).to(device).requires_grad_(True)
    z3_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(1),),
        requires_grad=True,
        device=device
    )
    optimizer_z1 = optim.Adam([z1_hat], lr=0.001)
    optimizer_others = optim.Adam([theta_hat, z2_hat, z3_hat], lr=0.01)
    
    best_loss = float('inf')
    patience_counter = 0
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        theta_hat_matrix = theta_hat.unsqueeze(1)
        z1_hat.data.clamp_(min=0.0, max=1.0)
        z1_hat_matrix = z1_hat.unsqueeze(0)
        z2_hat_matrix = z2_hat.unsqueeze(0)
        z3_hat_matrix = z3_hat.unsqueeze(0)
        prob_matrix = item_response_fn_3PL(z1_hat_matrix, z2_hat_matrix, z3_hat_matrix, theta_hat_matrix)
        assert prob_matrix.shape == response_matrix.shape
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]
        
        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer_z1.step()
        optimizer_others.step()
        optimizer_z1.zero_grad()
        optimizer_others.zero_grad()
        
        pbar.set_postfix({'loss': loss.item()})
        wandb.log({'loss': loss.item()})
        
        if abs(loss.item() - best_loss) > 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break

    return theta_hat, z1_hat, z2_hat, z3_hat

if __name__ == "__main__":
    wandb.init(project="mle_3pl_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    input_dir = '../data/pre_calibration/'
    output_dir = f'../data/mle_3pl_calibration/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    y = pd.read_csv(f'{input_dir}/{args.dataset}/matrix.csv', index_col=0).values
    theta_hat, z1_hat, z2_hat, z3_hat = mle_3pl_calibration(torch.tensor(y, dtype=torch.float32))
    
    z_df = pd.DataFrame({
        'z1': z1_hat.cpu().detach().numpy(),
        'z2': z2_hat.cpu().detach().numpy(),
        'z3': z3_hat.cpu().detach().numpy(),
    })
    z_df.to_csv(f"{output_dir}/z.csv", index=False)
    theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=["theta"])
    theta_df.to_csv(f"{output_dir}/theta.csv", index=False)
    