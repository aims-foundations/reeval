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
    optimizer_z1 = optim.Adam([z1_hat], lr=0.0001)
    optimizer_others = optim.Adam([theta_hat, z2_hat, z3_hat], lr=0.001)
    
    last_theta_hat = None
    last_z1_hat = None
    last_z2_hat = None
    last_z3_hat = None
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        if not (torch.isnan(theta_hat).any() or torch.isnan(z1_hat).any() \
            or torch.isnan(z2_hat).any() or torch.isnan(z3_hat).any()):
            last_theta_hat = theta_hat.cpu().detach().clone()
            last_z1_hat = z1_hat.cpu().detach().clone()
            last_z2_hat = z2_hat.cpu().detach().clone()
            last_z3_hat = z3_hat.cpu().detach().clone()
        else:
            break
        
        # theta_hat_norm = (theta_hat - theta_hat.mean()) / theta_hat.std()
        # z1_hat_norm = (z1_hat - z1_hat.mean()) / z1_hat.std()
        # z1_hat_norm.data.clamp_(min=0.0, max=1.0)
        # z2_hat_norm = (z2_hat - z2_hat.mean()) / z2_hat.std()
        # z3_hat_norm = (z3_hat - z3_hat.mean()) / z3_hat.std()

        theta_hat_matrix = theta_hat.unsqueeze(1)
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
        torch.nn.utils.clip_grad_value_([z1_hat, theta_hat, z2_hat, z3_hat], clip_value=1.0)
        
        print(f"Gradients:")
        for name, param in zip(['theta_hat', 'z1_hat', 'z2_hat', 'z3_hat'], [theta_hat, z1_hat, z2_hat, z3_hat]):
            print(f"{name} grad: {param.grad}")

            nan_indices = torch.nonzero(torch.isnan(param.grad)).squeeze()
            if nan_indices.numel() > 0:
                print(f"{name} grad NaN indices: {nan_indices.cpu().numpy()}")
        
        optimizer_z1.step()
        optimizer_others.step()
        optimizer_z1.zero_grad()
        optimizer_others.zero_grad()
        
        pbar.set_postfix({'loss': loss.item()})
        wandb.log({'loss': loss.item()})

    return last_theta_hat, last_z1_hat, last_z2_hat, last_z3_hat

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
    