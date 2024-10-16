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
    max_epoch: int = 3000,
    patience: int = 50,
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
    
    best_loss = float('inf')
    patience_counter = 0
    for _ in pbar:
        prob_matrix = item_response_fn_1PL_multi_dim(z_hat[None, :], theta_hat, a)
        assert prob_matrix.shape == response_matrix.shape

        mask = response_matrix != -1
        masked_response_matrix = response_matrix[mask]
        masked_prob_matrix = prob_matrix[mask]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item()})
        # wandb.log({'loss': loss.item()})

        if abs(loss.item() - best_loss) > 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
        
    return theta_hat, a, z_hat

if __name__ == "__main__":
    # wandb.init(project="mle_multi_dim_calibration")
    matrix_airbench = pd.read_csv('../data/pre_calibration/airbench/matrix.csv', index_col=0)
    matrix_mmlu = pd.read_csv('../data/pre_calibration/mmlu/matrix.csv', index_col=0)

    all_models = matrix_airbench.index.union(matrix_mmlu.index)
    all_questions = matrix_airbench.columns.union(matrix_mmlu.columns)
    combined_matrix = pd.DataFrame(-1, index=all_models, columns=all_questions)
    combined_matrix.update(matrix_airbench)
    combined_matrix.update(matrix_mmlu)
    print(combined_matrix.shape)
        
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
    