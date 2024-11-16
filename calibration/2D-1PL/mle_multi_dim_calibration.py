import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from utils import (
    DATASETS,
    error_bar_plot_single,
    goodness_of_fit_1PL_multi_dim_plot,
    item_response_fn_1PL_multi_dim,
    plot_hist,
    set_seed,
)


def mle_multi_dim_calibration(
    response_matrix: torch.Tensor,
    constraint: bool = True,
    dim: int = 2,
    max_epoch: int = 3000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)

    theta_hat = torch.normal(
        mean=0.0,
        std=1.0,
        size=(response_matrix.size(0), dim),
        requires_grad=True,
        device=device,
    )
    a = torch.normal(
        mean=0.0,
        std=1.0,
        size=(response_matrix.size(1), dim),
        requires_grad=True,
        device=device,
    )
    z_hat = torch.normal(
        mean=0.0,
        std=1.0,
        size=(response_matrix.size(1),),
        requires_grad=True,
        device=device,
    )

    optimizer = optim.Adam([theta_hat, a, z_hat], lr=0.01)

    last_theta_hat_norm = None
    if constraint:
        last_a_softmax = None
    else:
        last_a = None
    last_z_hat = None
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        if constraint:
            theta_hat_norm = (theta_hat - torch.mean(theta_hat)) / torch.std(theta_hat)
            a_softmax = torch.nn.functional.softmax(a, dim=1)
            prob_matrix = item_response_fn_1PL_multi_dim(
                z_hat[None, :], theta_hat_norm, a_softmax
            )
        # else:
        #     theta_hat = (theta_hat - torch.mean(theta_hat)) / torch.std(theta_hat)
        #     prob_matrix = item_response_fn_1PL_multi_dim(z_hat[None, :], theta_hat, a)
        assert prob_matrix.shape == response_matrix.shape

        mask = response_matrix != -1
        masked_response_matrix = response_matrix[mask]
        masked_prob_matrix = prob_matrix[mask]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({"loss": loss.item()})
        # wandb.log({'loss': loss.item()})

        if constraint:
            if not (
                torch.isnan(theta_hat_norm).any()
                or torch.isnan(a_softmax).any()
                or torch.isnan(z_hat).any()
            ):
                last_theta_hat_norm = theta_hat_norm.cpu().detach().clone()
                last_a_softmax = a_softmax.cpu().detach().clone()
                last_z_hat = z_hat.cpu().detach().clone()
            else:
                break
        # else:
        #     if not (torch.isnan(theta_hat).any() or torch.isnan(a).any() or torch.isnan(z_hat).any()):
        #         last_theta_hat = theta_hat.cpu().detach().clone()
        #         last_a = a.cpu().detach().clone()
        #         last_z_hat = z_hat.cpu().detach().clone()
        #     else:
        #         break

    if constraint:
        return last_theta_hat_norm, last_a_softmax, last_z_hat
    # else:
    #     return last_theta_hat, last_a, last_z_hat


if __name__ == "__main__":
    # wandb.init(project="mle_multi_dim_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--constraint", type=str, required=True, choices=["True", "False"]
    )
    args = parser.parse_args()

    if args.constraint == "True":
        args.constraint = True
    elif args.constraint == "False":
        args.constraint = False

    set_seed(42)
    output_dir = f"../data/mle_multi_dim_calibration"
    plot_dir = f"../plot/mle_multi_dim_calibration"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # combined_matrix = pd.DataFrame()
    # for dataset in DATASETS:
    #     matrix = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
    #     if combined_matrix.empty:
    #         combined_matrix = matrix
    #     else:
    #         combined_matrix = combined_matrix.join(matrix, how='outer', rsuffix='_dup')
    # combined_matrix.fillna(-1, inplace=True)
    # print(combined_matrix.shape)
    # combined_matrix.to_csv(f"{output_dir}/combined_matrix.csv")
    combined_matrix = pd.read_csv(f"{output_dir}/combined_matrix.csv", index_col=0)

    theta_hat, a, z_hat = mle_multi_dim_calibration(
        response_matrix=torch.tensor(combined_matrix.values, dtype=torch.float32),
        constraint=args.constraint,
    )
    z_df = pd.DataFrame(z_hat.cpu().detach().numpy(), columns=["z"])
    z_df.to_csv(f"{output_dir}/z_con_{args.constraint}.csv", index=False)
    a_df = pd.DataFrame(
        a.cpu().detach().numpy(), columns=[f"a_{i}" for i in range(a.size(1))]
    )
    a_df.to_csv(f"{output_dir}/a_con_{args.constraint}.csv", index=False)
    theta_df = pd.DataFrame(
        theta_hat.cpu().detach().numpy(),
        columns=[f"theta_{i}" for i in range(theta_hat.size(1))],
    )
    theta_df.to_csv(f"{output_dir}/theta_con_{args.constraint}.csv", index=False)

    # theta_hat = pd.read_csv(f"{output_dir}/theta_con_{args.constraint}.csv")
    # a = pd.read_csv(f"{output_dir}/a_con_{args.constraint}.csv")
    # z_hat = pd.read_csv(f"{output_dir}/z_con_{args.constraint}.csv")

    gof_means, gof_stds = [], []
    a_means = []
    for dataset in tqdm(DATASETS):
        matrix = pd.read_csv(
            f"../data/pre_calibration/{dataset}/matrix.csv", index_col=0
        )
        response_matrix = combined_matrix.loc[matrix.index, matrix.columns].values
        row_indices = [combined_matrix.index.get_loc(i) for i in matrix.index]
        col_indices = [combined_matrix.columns.get_loc(i) for i in matrix.columns]

        theta_hat_subset = theta_hat[row_indices].cpu().detach()
        z_hat_subset = z_hat[col_indices].cpu().detach()
        a_subset = a[col_indices].cpu().detach()

        gof_mean, gof_std = goodness_of_fit_1PL_multi_dim_plot(
            z=z_hat_subset,
            theta=theta_hat_subset,
            a=a_subset,
            y=torch.tensor(response_matrix, dtype=torch.float32),
            plot_path=f"{plot_dir}/goodness_of_fit_con_{args.constraint}_{dataset}.png",
        )
        gof_means.append(gof_mean)
        gof_stds.append(gof_std)

        if args.constraint:
            a_mean = a_subset[:, 0].numpy().mean()
            a_means.append(a_mean)
            # plot_hist(
            #     data=a_subset[:, 0].numpy(),
            #     plot_path=f'{plot_dir}/a_histogram_{dataset}.png',
            #     ylabel='Histiogram of a',
            # )

    gof_df = pd.DataFrame(
        {
            "datasets": DATASETS,
            "gof_means": gof_means,
            "gof_stds": gof_stds,
        }
    )
    gof_df.to_csv(f"{plot_dir}/dim2_1pl_gof_con_{args.constraint}.csv", index=False)

    error_bar_plot_single(
        datasets=DATASETS,
        means=gof_means,
        stds=gof_stds,
        plot_path=f"{plot_dir}/mle_multi_dim_calibration_summarize_gof_con_{args.constraint}",
        xlabel=r"Goodness of Fit",
    )

    if args.constraint:
        error_bar_plot_single(
            datasets=DATASETS,
            means=a_means,
            stds=[0] * len(a_means),
            plot_path=f"{plot_dir}/mle_multi_dim_calibration_summarize_a_con_{args.constraint}",
            xlabel=r"Mean of $a$",
        )
