import os
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from utils import item_response_fn_1PL_multi_dim, set_seed, goodness_of_fit_1PL_multi_dim_plot
import torch.optim as optim

def mle_multi_dim_calibration(
    response_matrix: torch.Tensor,
    dim: int=2,
    max_epoch: int = 3000,
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
    
    last_theta_hat = None
    last_a = None
    last_z_hat = None
    pbar = tqdm(range(max_epoch))
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
        
        if not (torch.isnan(theta_hat).any() or torch.isnan(a).any() or torch.isnan(z_hat).any()):
            last_theta_hat = theta_hat.cpu().detach().clone()
            last_a = a.cpu().detach().clone()
            last_z_hat = z_hat.cpu().detach().clone()
        else:
            break
        
    return last_theta_hat, last_a, last_z_hat

if __name__ == "__main__":
    # wandb.init(project="mle_multi_dim_calibration")
    set_seed(42)
    output_dir = f'../data/mle_multi_dim_calibration'
    plot_dir = f'../plot/mle_multi_dim_calibration'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    matrix_airbench = pd.read_csv('../data/pre_calibration/airbench/matrix.csv', index_col=0)
    matrix_mmlu = pd.read_csv('../data/pre_calibration/mmlu/matrix.csv', index_col=0)

    all_models = matrix_airbench.index.union(matrix_mmlu.index)
    all_questions = matrix_airbench.columns.union(matrix_mmlu.columns)
    combined_matrix = pd.DataFrame(-1, index=all_models, columns=all_questions)
    combined_matrix.update(matrix_airbench)
    combined_matrix.update(matrix_mmlu)
    print(combined_matrix.shape)
    combined_matrix_df = pd.DataFrame(combined_matrix.values, index=all_models, columns=all_questions)
    combined_matrix_df.to_csv(f"{output_dir}/combined_matrix.csv")
    
    theta_hat, a, z_hat = mle_multi_dim_calibration(
        torch.tensor(combined_matrix.values, dtype=torch.float32),
    )
    z_df = pd.DataFrame(z_hat.cpu().detach().numpy(), columns=["z"])
    z_df.to_csv(f"{output_dir}/z.csv", index=False)
    a_df = pd.DataFrame(a.cpu().detach().numpy(), columns=[f"a_{i}" for i in range(a.size(1))])
    a_df.to_csv(f"{output_dir}/a.csv", index=False)
    theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=[f"theta_{i}" for i in range(theta_hat.size(1))])
    theta_df.to_csv(f"{output_dir}/theta.csv", index=False)

    response_matrix_airbench = combined_matrix.loc[matrix_airbench.index, matrix_airbench.columns].values
    airbench_row_indices = [combined_matrix.index.get_loc(i) for i in matrix_airbench.index]
    airbench_col_indices = [combined_matrix.columns.get_loc(i) for i in matrix_airbench.columns]
    response_tensor_airbench = torch.tensor(response_matrix_airbench, dtype=torch.float32)
    theta_hat_airbench = theta_hat[airbench_row_indices].cpu().detach()
    z_hat_airbench = z_hat[airbench_col_indices].cpu().detach()
    a_airbench = a[airbench_col_indices].cpu().detach()
    mean_diff_airbench, std_diff_airbench = goodness_of_fit_1PL_multi_dim_plot(
        z=z_hat_airbench, 
        theta=theta_hat_airbench,
        a=a_airbench,
        y=response_tensor_airbench,
        plot_path=f'{plot_dir}/goodness_of_fit_airbench.png'
    )

    response_matrix_mmlu = combined_matrix.loc[matrix_mmlu.index, matrix_mmlu.columns].values
    mmlu_row_indices = [combined_matrix.index.get_loc(i) for i in matrix_mmlu.index]
    mmlu_col_indices = [combined_matrix.columns.get_loc(i) for i in matrix_mmlu.columns]
    response_tensor_mmlu = torch.tensor(response_matrix_mmlu, dtype=torch.float32)
    theta_hat_mmlu = theta_hat[mmlu_row_indices].cpu().detach()
    z_hat_mmlu = z_hat[mmlu_col_indices].cpu().detach()
    a_mmlu = a[mmlu_col_indices].cpu().detach()
    mean_diff_mmlu, std_diff_mmlu = goodness_of_fit_1PL_multi_dim_plot(
        z=z_hat_mmlu,
        theta=theta_hat_mmlu,
        a=a_mmlu,
        y=response_tensor_mmlu,
        plot_path=f'{plot_dir}/goodness_of_fit_mmlu.png'
    )