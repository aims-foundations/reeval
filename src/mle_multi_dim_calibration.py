import argparse
import os
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from utils import item_response_fn_1PL_multi_dim, set_seed
import torch.optim as optim

def mle_multi_dim_calibration(
    response_matrix: torch.Tensor,
    dim: int=2,
    max_epoch: int = 3000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)

    theta_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(0), dim),
        requires_grad=True, device=device
    )
    a = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(1), dim),
        requires_grad=True, device=device
    )
    z_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(1),),
        requires_grad=True, device=device
    )

    optimizer = optim.Adam([theta_hat, a, z_hat], lr=0.01)
    pbar = tqdm(range(max_epoch))

    for _ in pbar:
        prob_matrix = item_response_fn_1PL_multi_dim(
            z_hat[None, :], theta_hat, a
        )
        assert prob_matrix.shape == response_matrix.shape

        mask = response_matrix != -1
        masked_response_matrix = response_matrix[mask]
        masked_prob_matrix = prob_matrix[mask]
        print(masked_prob_matrix)

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"theta max: {theta_hat.max()}, min: {theta_hat.min()}")
        print(f"a max: {a.max()}, min: {a.min()}")
        print(f"z max: {z_hat.max()}, min: {z_hat.min()}")
        pbar.set_postfix({'loss': loss.item()})
        # wandb.log({'loss': loss.item()})

    return theta_hat, a, z_hat

if __name__ == "__main__":
    # wandb.init(project="mle_multi_dim_calibration")
    matrix_raft = pd.read_csv('../data/pre_calibration/raft/matrix.csv', index_col=0)
    matrix_entity = pd.read_csv('../data/pre_calibration/ent_data/matrix.csv', index_col=0)

    all_models = matrix_raft.index.union(matrix_entity.index)
    all_questions = matrix_raft.columns.union(matrix_entity.columns)
    combined_matrix = pd.DataFrame(-1, index=all_models, columns=all_questions)
    combined_matrix.update(matrix_raft)
    combined_matrix.update(matrix_entity)
        
    set_seed(42)
    output_dir = f'../data/mle_multi_dim_calibration'
    os.makedirs(output_dir, exist_ok=True)
    
    theta_hat, a, z_hat = mle_multi_dim_calibration(torch.tensor(combined_matrix.values, dtype=torch.float32))
    
    z_df = pd.DataFrame(z_hat.cpu().detach().numpy(), columns=["z"])
    z_df.to_csv(f"{output_dir}/z.csv", index=False)
    a_df = pd.DataFrame(a.cpu().detach().numpy(), columns=[f"a_{i}" for i in range(a.size(1))])
    a_df.to_csv(f"{output_dir}/a.csv", index=False)
    theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=[f"theta_{i}" for i in range(theta_hat.size(1))])
    theta_df.to_csv(f"{output_dir}/theta.csv", index=False)
    