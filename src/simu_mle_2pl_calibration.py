import argparse
import os
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from utils import item_response_fn_2PL, set_seed, goodness_of_fit_2PL_plot, theta_corr_ctt_plot
import torch.optim as optim

def simu_mle_2pl_calibration(
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
    z2_hat = torch.distributions.LogNormal(0.0, 0.5).sample(
        (response_matrix.size(1),)
    ).to(device).requires_grad_(True)
    z3_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(1),),
        requires_grad=True,
        device=device
    )
    optimizer = optim.Adam([theta_hat, z2_hat, z3_hat], lr=0.01)
    
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        theta_hat_matrix = theta_hat.unsqueeze(1)
        z2_hat_matrix = z2_hat.unsqueeze(0)
        z3_hat_matrix = z3_hat.unsqueeze(0)
        prob_matrix = item_response_fn_2PL(z2_hat_matrix, z3_hat_matrix, theta_hat_matrix)
        assert prob_matrix.shape == response_matrix.shape
        
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
        
    return theta_hat, z2_hat, z3_hat

if __name__ == "__main__":
    # wandb.init(project="simu_mle_2pl_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    input_dir = '../data/pre_calibration/'
    output_dir = f'../data/simu_mle_2pl_calibration/{args.dataset}'
    plot_dir = f'../plot/simu_mle_2pl_calibration/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    y = pd.read_csv(f'{input_dir}/{args.dataset}/matrix.csv', index_col=0).values
    theta_hat, z2_hat, z3_hat = simu_mle_2pl_calibration(torch.tensor(y, dtype=torch.float64))
    
    z_df = pd.DataFrame({
        'z2': z2_hat.cpu().detach().numpy(),
        'z3': z3_hat.cpu().detach().numpy(),
    })
    z_df.to_csv(f"{output_dir}/z.csv", index=False)
    theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=["theta"])
    theta_df.to_csv(f"{output_dir}/theta.csv", index=False)

    _, _ = goodness_of_fit_2PL_plot(
        z2=z2_hat.clone().detach(),        
        z3=z3_hat.clone().detach(),                        
        theta=theta_hat.clone().detach(),
        y=torch.tensor(y, dtype=torch.float64),
        plot_path=f"{plot_dir}/goodness_of_fit_{args.dataset}.png"
    )
    
    _, _ = theta_corr_ctt_plot(
        theta=theta_hat.cpu().detach().numpy(),
        y=y,
        plot_path=f"{plot_dir}/theta_corr_ctt_{args.dataset}",
    )
    