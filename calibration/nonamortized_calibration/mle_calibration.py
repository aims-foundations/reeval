import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from utils.utils import IRT, set_seed


def nonamor_calibration(
    response_matrix: torch.Tensor, max_epoch: int = 3000, D=1, PL=1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    n_models, n_questions = response_matrix.shape

    irt_model = IRT(n_questions, n_models, D, PL)
    irt_model = irt_model.to(device)

    optimizer = optim.Adam(irt_model.parameters(), lr=0.01)

    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        prob_matrix = irt_model.forward()

        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        irt_model.normalize()

        pbar.set_postfix({"loss": loss.item()})

    theta_hat = irt_model.get_ability()
    z_hat = irt_model.get_difficulty()

    return theta_hat, z_hat


if __name__ == "__main__":
    wandb.init(project="nonamor_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    args = parser.parse_args()

    set_seed(42)
    input_dir = "../../data/pre_calibration"
    output_dir = f"../../data/nonamor_calibration/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    y = pd.read_csv(f"{input_dir}/{args.dataset}/matrix.csv", index_col=0).values
    theta_hat, z_hat = nonamor_calibration(
        torch.tensor(y, dtype=torch.float32), D=args.D, PL=args.PL
    )

    z_df = pd.DataFrame(z_hat.cpu().detach().numpy(), columns=["z"])
    z_df.to_csv(f"{output_dir}/nonamor_z.csv", index=False)

    if args.D == 1:
        theta_df = pd.DataFrame(theta_hat.cpu().detach().numpy(), columns=["theta"])
    else:
        theta_df = pd.DataFrame(
            theta_hat.cpu().detach().numpy(),
            columns=[f"theta_{i}" for i in range(args.D)],
        )
    theta_df.to_csv(f"{output_dir}/nonamor_theta.csv", index=False)
