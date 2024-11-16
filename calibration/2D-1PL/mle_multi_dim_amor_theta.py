import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
plt.style.use("seaborn-v0_8-paper")
from utils import (
    DATASETS,
    error_bar_plot_double,
    goodness_of_fit_1PL_multi_dim_plot,
    item_response_fn_1PL_multi_dim,
    set_seed,
)


def mle_multi_dim_amor_theta(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    constraint: bool,
    feat_train: torch.Tensor,
    feat_test: torch.Tensor,
    dim: int = 2,
    max_epoch: int = 3000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    num_model, num_item = y_train.shape
    feat_train = feat_train[:, None].to(device)
    feat_test = feat_test[:, None].to(device)

    W = torch.normal(
        mean=0.0,
        std=1.0,
        size=(feat_train.shape[1], dim),
        requires_grad=True,
        device=device,
    )
    b = torch.normal(mean=0.0, std=1.0, size=(dim,), requires_grad=True, device=device)
    a = torch.normal(
        mean=0.0, std=1.0, size=(num_item, dim), requires_grad=True, device=device
    )
    z_hat = torch.normal(
        mean=0.0, std=1.0, size=(num_item,), requires_grad=True, device=device
    )

    optimizer = optim.Adam([W, b, a, z_hat], lr=0.01)

    last_W = None
    last_b = None
    if constraint:
        last_a_softmax = None
    # else:
    #     last_a = None
    last_z_hat = None
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        # if constraint:
        # else:
        #     prob_train = item_response_fn_1PL_multi_dim(z_hat[None, :], theta_train, a)
        theta_train = torch.mm(feat_train, W) + b  # (num_model, dim=2)
        theta_train_norm = (theta_train - torch.mean(theta_train)) / torch.std(
            theta_train
        )
        a_softmax = torch.nn.functional.softmax(a, dim=1)
        prob_train = item_response_fn_1PL_multi_dim(
            z_hat[None, :], theta_train_norm, a_softmax
        )
        assert prob_train.shape == y_train.shape
        mask_train = y_train != -1
        masked_y_train = y_train[mask_train]
        masked_prob_train = prob_train[mask_train]

        train_loss = (
            -torch.distributions.Bernoulli(masked_prob_train)
            .log_prob(masked_y_train)
            .mean()
        )
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            W_temp = W.clone().detach()
            theta_test = torch.mm(feat_test, W_temp) + b
            theta_test_norm = (theta_test - torch.mean(theta_test)) / torch.std(
                theta_test
            )
            a_softmax_temp = a_softmax.clone().detach()
            prob_test = item_response_fn_1PL_multi_dim(
                z_hat[None, :], theta_test_norm, a_softmax_temp
            )
            assert prob_test.shape == y_test.shape
            mask_test = y_test != -1
            masked_y_test = y_test[mask_test]
            masked_prob_test = prob_test[mask_test]
            test_loss = (
                -torch.distributions.Bernoulli(masked_prob_test)
                .log_prob(masked_y_test)
                .mean()
            )

        pbar.set_postfix(
            {"train_loss": train_loss.item(), "test_loss": test_loss.item()}
        )
        wandb.log({"train_loss": train_loss.item(), "test_loss": test_loss.item()})

        if not (
            torch.isnan(W).any()
            or torch.isnan(b).any()
            or torch.isnan(a_softmax).any()
            or torch.isnan(z_hat).any()
        ):
            last_W = W.cpu().detach().clone()
            last_b = b.cpu().detach().clone()
            last_a_softmax = a_softmax.cpu().detach().clone()
            last_z_hat = z_hat.cpu().detach().clone()
        else:
            break
        # if constraint:
        # else:
        #     if not (torch.isnan(theta_train).any() or torch.isnan(a).any() or torch.isnan(z_hat).any()):
        #         last_theta_train = theta_train.cpu().detach().clone()
        #         last_a = a.cpu().detach().clone()
        #         last_z_hat = z_hat.cpu().detach().clone()
        #     else:
        #         break

    return last_W, last_b, last_a_softmax, last_z_hat
    # if constraint:
    # else:
    #     return last_theta_train, last_a, last_z_hat


if __name__ == "__main__":
    wandb.init(project="mle_multi_dim_amor_theta")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--constraint", type=str, default="True", choices=["True", "False"]
    )
    args = parser.parse_args()

    if args.constraint == "True":
        args.constraint = True
    # elif args.constraint == 'False':
    #     args.constraint = False

    set_seed(42)
    output_dir = f"../data/mle_multi_dim_amor_theta"
    plot_dir = f"../plot/mle_multi_dim_amor_theta"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    model_id_df = pd.read_csv("configs/model_id_final.csv")
    valid_model_names = model_id_df["model_names_reeval"].values
    feat_matrix = model_id_df["FLOPs (1E21)"].values
    feat_matrix = np.log(feat_matrix)
    # feat_matrix = (feat_matrix - feat_matrix.mean(axis=0)) / feat_matrix.std(axis=0)

    # valid_model_names_test = ['meta_llama-2-70b', 'meta_llama-2-7b', 'meta_llama-2-13b']
    test_size = int(len(valid_model_names) * 0.2)
    valid_model_names_test = list(
        np.random.choice(valid_model_names, size=test_size, replace=False)
    )
    print(valid_model_names_test)
    valid_model_indices_test = [
        i for i, name in enumerate(valid_model_names) if name in valid_model_names_test
    ]
    feat_matrix_test = feat_matrix[valid_model_indices_test]
    valid_model_names_train = [
        name for name in valid_model_names if name not in valid_model_names_test
    ]
    valid_model_indices_train = [
        i for i, name in enumerate(valid_model_names) if name in valid_model_names_train
    ]
    feat_matrix_train = feat_matrix[valid_model_indices_train]

    # valid_datasets = []
    # combined_matrix_train = pd.DataFrame(index=valid_model_names_train)
    # combined_matrix_test = pd.DataFrame(index=valid_model_names_test)
    # for dataset in DATASETS:
    #     matrix = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)

    #     filtered_matrix_train = matrix[matrix.index.isin(valid_model_names_train)]
    #     if not filtered_matrix_train.empty:
    #         valid_datasets.append(dataset)
    #         combined_matrix_train = combined_matrix_train.join(filtered_matrix_train, how='outer', rsuffix='_dup')
    #         # print(f"Dataset: {dataset}, left model num: {filtered_matrix_train.shape[0]}, left models: {filtered_matrix_train.index.tolist()}")

    #         filtered_matrix_test = matrix[matrix.index.isin(valid_model_names_test)]
    #         combined_matrix_test = combined_matrix_test.join(filtered_matrix_test, how='outer', rsuffix='_dup')

    # valid_datasets_df = pd.DataFrame(valid_datasets, columns=["dataset"])
    # valid_datasets_df.to_csv(f"{output_dir}/valid_datasets.csv", index=False)

    # combined_matrix_train.fillna(-1, inplace=True)
    # combined_matrix_train = combined_matrix_train.reindex(valid_model_names_train)
    # print(combined_matrix_train.shape)
    # assert combined_matrix_train.index.tolist() == valid_model_names_train
    # combined_matrix_train.to_csv(f"{output_dir}/combined_matrix_train.csv")

    # combined_matrix_test.fillna(-1, inplace=True)
    # combined_matrix_test = combined_matrix_test.reindex(valid_model_names_test)
    # print(combined_matrix_test.shape)
    # assert combined_matrix_test.index.tolist() == valid_model_names_test, f"{combined_matrix_test.index.tolist()} != {valid_model_names_test}"
    # combined_matrix_test.to_csv(f"{output_dir}/combined_matrix_test.csv")

    valid_datasets = pd.read_csv(f"{output_dir}/valid_datasets.csv").values.flatten()
    combined_matrix_train = pd.read_csv(
        f"{output_dir}/combined_matrix_train.csv", index_col=0
    )
    combined_matrix_test = pd.read_csv(
        f"{output_dir}/combined_matrix_test.csv", index_col=0
    )

    # W, b, a, z_hat = mle_multi_dim_amor_theta(
    #     y_train=torch.tensor(combined_matrix_train.values, dtype=torch.float32),
    #     y_test=torch.tensor(combined_matrix_test.values, dtype=torch.float32),
    #     constraint=args.constraint,
    #     feat_train=torch.tensor(feat_matrix_train, dtype=torch.float32),
    #     feat_test=torch.tensor(feat_matrix_test, dtype=torch.float32),
    # )
    # z_df = pd.DataFrame(z_hat.cpu().detach().numpy(), columns=["z"])
    # z_df.to_csv(f"{output_dir}/z_con_{args.constraint}.csv", index=False)
    # W = W.cpu().detach().numpy()
    # b = b.cpu().detach().numpy()
    # a = a.cpu().detach().numpy()
    # np.save(f"{output_dir}/W_con_{args.constraint}.npy", W)
    # np.save(f"{output_dir}/b_con_{args.constraint}.npy", b)
    # np.save(f"{output_dir}/a_con_{args.constraint}.npy", a)

    z_hat = pd.read_csv(f"{output_dir}/z_con_{args.constraint}.csv").values
    W = np.load(f"{output_dir}/W_con_{args.constraint}.npy")
    b = np.load(f"{output_dir}/b_con_{args.constraint}.npy")
    a = np.load(f"{output_dir}/a_con_{args.constraint}.npy")

    theta_gt_names = pd.read_csv(
        "../data/mle_multi_dim_calibration/combined_matrix.csv", index_col=0
    ).index.to_list()
    theta_gt_all = pd.read_csv("../data/mle_multi_dim_calibration/theta.csv").values
    theta_train_gt_indices = [
        i for i, name in enumerate(theta_gt_names) if name in valid_model_names_train
    ]
    theta_train_gt = theta_gt_all[theta_train_gt_indices]
    theta_test_gt_indices = [
        i for i, name in enumerate(theta_gt_names) if name in valid_model_names_test
    ]
    theta_test_gt = theta_gt_all[theta_test_gt_indices]
    theta_gt_indices = [
        i for i, name in enumerate(theta_gt_names) if name in valid_model_names
    ]
    theta_gt = theta_gt_all[theta_gt_indices]

    theta_train_pred = feat_matrix_train[:, None] @ W + b
    theta_test_pred = feat_matrix_test[:, None] @ W + b
    assert theta_train_gt.shape == theta_train_pred.shape
    assert theta_test_gt.shape == theta_test_pred.shape
    assert feat_matrix.shape[0] == theta_gt.shape[0]
    assert feat_matrix_train.shape[0] == theta_train_gt.shape[0]
    assert feat_matrix_test.shape[0] == theta_test_gt.shape[0]

    feat_matrix_df = pd.DataFrame(
        feat_matrix, index=valid_model_names, columns=["feat"]
    )
    feat_matrix_df.to_csv(f"{output_dir}/feat_matrix.csv")
    feat_matrix_train_df = pd.DataFrame(
        feat_matrix_train, index=valid_model_names_train, columns=["feat"]
    )
    feat_matrix_train_df.to_csv(f"{output_dir}/feat_matrix_train.csv")
    feat_matrix_test_df = pd.DataFrame(
        feat_matrix_test, index=valid_model_names_test, columns=["feat"]
    )
    feat_matrix_test_df.to_csv(f"{output_dir}/feat_matrix_test.csv")
    theta_train_gt_df = pd.DataFrame(
        theta_train_gt, index=valid_model_names_train, columns=["theta_0", "theta_1"]
    )
    theta_train_gt_df.to_csv(f"{output_dir}/theta_train_gt.csv")
    theta_test_gt_df = pd.DataFrame(
        theta_test_gt, index=valid_model_names_test, columns=["theta_0", "theta_1"]
    )
    theta_test_gt_df.to_csv(f"{output_dir}/theta_test_gt.csv")
    theta_train_pred_df = pd.DataFrame(
        theta_train_pred, index=valid_model_names_train, columns=["theta_0", "theta_1"]
    )
    theta_train_pred_df.to_csv(f"{output_dir}/theta_train_pred.csv")
    theta_test_pred_df = pd.DataFrame(
        theta_test_pred, index=valid_model_names_test, columns=["theta_0", "theta_1"]
    )
    theta_test_pred_df.to_csv(f"{output_dir}/theta_test_pred.csv")

    x = np.linspace(0, feat_matrix.max() + 5, 100)
    y = x[:, None] @ W + b
    assert x.shape[0] == y.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(
        feat_matrix, theta_gt[:, 0], label="Non-amortized", color="black", alpha=0.5
    )
    ax1.scatter(
        feat_matrix_train,
        theta_train_pred[:, 0],
        label="Amortized train",
        color="blue",
        alpha=0.5,
    )
    ax1.scatter(
        feat_matrix_test,
        theta_test_pred[:, 0],
        label="Amortized test",
        color="red",
        alpha=0.5,
    )
    ax1.plot(x, y[:, 0], color="blue", alpha=0.5)
    ax1.set_title(r"$\theta_0$", fontsize=25)
    ax1.tick_params(axis="both", labelsize=25)
    ax1.legend(fontsize=10)

    ax2.scatter(
        feat_matrix, theta_gt[:, 1], label="Non-amortized", color="black", alpha=0.5
    )
    ax2.scatter(
        feat_matrix_train,
        theta_train_pred[:, 1],
        label="Amortized train",
        color="blue",
        alpha=0.5,
    )
    ax2.scatter(
        feat_matrix_test,
        theta_test_pred[:, 1],
        label="Amortized test",
        color="red",
        alpha=0.5,
    )
    ax2.plot(x, y[:, 1], color="blue", alpha=0.5)
    ax2.set_title(r"$\theta_1$", fontsize=25)
    ax2.tick_params(axis="both", labelsize=25)

    plt.legend(fontsize=10)
    plt.savefig(f"{plot_dir}/theta_to_feat.png", dpi=300, bbox_inches="tight")
    plt.close()

    gof_mean_trains, gof_std_trains = [], []
    gof_mean_tests, gof_std_tests = [], []
    for dataset in tqdm(valid_datasets):
        matrix = pd.read_csv(
            f"../data/pre_calibration/{dataset}/matrix.csv", index_col=0
        )
        col_indices = [combined_matrix_train.columns.get_loc(i) for i in matrix.columns]
        z_hat_subset = z_hat[col_indices]
        a_subset = a[col_indices]

        matrix_train = matrix[matrix.index.isin(valid_model_names_train)]
        matrix_test = matrix[matrix.index.isin(valid_model_names_test)]
        train_indices = [
            i
            for i, name in enumerate(valid_model_names_train)
            if name in matrix_train.index
        ]
        test_indices = [
            i
            for i, name in enumerate(valid_model_names_test)
            if name in matrix_test.index
        ]
        feat_train = feat_matrix_train[train_indices]
        feat_test = feat_matrix_test[test_indices]
        theta_train = feat_train[:, None] @ W + b
        theta_test = feat_test[:, None] @ W + b

        gof_mean_train, gof_std_train = goodness_of_fit_1PL_multi_dim_plot(
            z=torch.tensor(z_hat_subset, dtype=torch.float32),
            theta=torch.tensor(theta_train, dtype=torch.float32),
            a=torch.tensor(a_subset, dtype=torch.float32),
            y=torch.tensor(matrix_train.values, dtype=torch.float32),
            plot_path=f"{plot_dir}/goodness_of_fit_con_{args.constraint}_{dataset}_train.png",
        )
        gof_mean_trains.append(gof_mean_train)
        gof_std_trains.append(gof_std_train)

        gof_mean_test, gof_std_test = goodness_of_fit_1PL_multi_dim_plot(
            z=torch.tensor(z_hat_subset, dtype=torch.float32),
            theta=torch.tensor(theta_test, dtype=torch.float32),
            a=torch.tensor(a_subset, dtype=torch.float32),
            y=torch.tensor(matrix_test.values, dtype=torch.float32),
            plot_path=f"{plot_dir}/goodness_of_fit_con_{args.constraint}_{dataset}_test.png",
        )
        gof_mean_tests.append(gof_mean_test)
        gof_std_tests.append(gof_std_test)

    gof_train_df = pd.DataFrame(
        {
            "datasets": valid_datasets,
            "gof_means": gof_mean_trains,
            "gof_stds": gof_std_trains,
        }
    )
    gof_train_df.to_csv(
        f"{plot_dir}/mle_multi_dim_amor_theta_gof_con_{args.constraint}_train.csv",
        index=False,
    )
    gof_test_df = pd.DataFrame(
        {
            "datasets": valid_datasets,
            "gof_means": gof_mean_tests,
            "gof_stds": gof_std_tests,
        }
    )
    gof_test_df.to_csv(
        f"{plot_dir}/mle_multi_dim_amor_theta_gof_con_{args.constraint}_test.csv",
        index=False,
    )

    error_bar_plot_double(
        datasets=valid_datasets,
        means_train=gof_mean_trains,
        stds_train=gof_std_trains,
        means_test=gof_mean_tests,
        stds_test=gof_std_tests,
        plot_path=f"{plot_dir}/mle_multi_dim_amor_theta_summarize_gof_con_{args.constraint}",
        xlabel=r"Goodness of Fit",
        plot_std=False,
    )
