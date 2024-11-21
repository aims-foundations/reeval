import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
plt.style.use("seaborn-v0_8-paper")

from .constants import HELM_MODEL_MAP
from .irt import IRT


def arg2str(args):
    return (
        (
            f"{args.dataset}/"
            f"s{args.seed}_"
            f"{args.fitting_method}_{args.PL}pl_{args.D}"
            f"d{'_aq' if args.amortized_question else ''}"
            f"{'_as' if args.amortized_student else ''}"
        )
        + (f"_nl{args.n_layers}" if args.n_layers is not None else "")
        + (f"_hd{args.hidden_dim}" if args.hidden_dim is not None else "")
    )


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(length):
    indices = np.arange(length)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return train_indices.tolist(), test_indices.tolist()


def inverse_sigmoid(x):
    epsilon = 1e-7
    x = torch.clamp(x, min=epsilon, max=1 - epsilon)  # Clip the input to (0, 1)
    return torch.log(x / (1 - x))


def goodness_of_fit(
    item_parms: torch.Tensor,  # n_items x n_item_parameters
    theta: torch.Tensor,  # n_testtakers x n_testtaker_parameters
    y: torch.Tensor,
    num_bins: int = 6,
):
    n_items = item_parms.shape[0]
    n_testtakers = theta.shape[0]

    assert y.shape[0] == n_testtakers, f"{y.shape[0]} != {n_testtakers}"
    assert y.shape[1] == n_items, f"{y.shape[1]} != {n_items}"

    bins_start, bins_end = (
        torch.min(theta, dim=0).values,
        torch.max(theta, dim=0).values,
    )
    bins = torch.stack(
        [
            torch.linspace(bin_start, bin_end, num_bins + 1).to(theta)
            for bin_start, bin_end in zip(bins_start, bins_end)
        ],
        dim=-1,
    )
    # >>> n_bins x D

    diffs = []
    thetas_mid = (bins[:-1] + bins[1:]) / 2
    item_parms_list = [*(item_parms[..., 0:3].T), item_parms[..., 3:]]
    probs_theoretical = IRT.compute_prob(thetas_mid, *item_parms_list)

    bin_masks = (theta[:, None] >= bins[:-1]) & (theta[:, None] < bins[1:])
    D = theta.shape[1]

    for d in range(D):
        diff_D = []
        for bi in range(num_bins):
            bin_mask = bin_masks[:, bi, d]
            if bin_mask.sum() <= 0:
                continue

            y_bins = y[bin_mask]
            y_bins[y_bins == -1] = torch.nan
            prob_empirical = y_bins.nanmean(dim=0)

            nan_mask = torch.isnan(prob_empirical)
            prob_theoretical = probs_theoretical[bi]

            diff = 1 - torch.abs(prob_empirical - prob_theoretical)[~nan_mask]
            diff_D.extend(diff.tolist())

        diffs.append(np.array(diff_D))
    return np.concatenate(diffs)


def goodness_of_fit_1PL_multi_dim(
    z: torch.Tensor,
    theta: torch.Tensor,
    a: torch.Tensor,
    y: torch.Tensor,
    bin_size: int = 6,
):
    assert y.shape[1] == z.shape[0], f"{y.shape[1]} != {z.shape[0]}"
    assert y.shape[0] == theta.shape[0], f"{y.shape[0]} != {theta.shape[0]}"

    bin_start_dim1, bin_end_dim1 = torch.min(theta[:, 0]), torch.max(theta[:, 0])
    bins_dim1 = torch.linspace(bin_start_dim1, bin_end_dim1, bin_size + 1)
    print(bins_dim1)
    bin_start_dim2, bin_end_dim2 = torch.min(theta[:, 1]), torch.max(theta[:, 1])
    bins_dim2 = torch.linspace(bin_start_dim2, bin_end_dim2, bin_size + 1)
    print(bins_dim2)

    diff_list = []
    for i in range(z.shape[0]):
        single_z = z[i]
        single_a = a[i]
        y_col = y[:, i]

        for j in range(bins_dim1.shape[0] - 1):
            for k in range(bins_dim2.shape[0] - 1):
                bin_mask = (
                    (theta[:, 0] >= bins_dim1[j])
                    & (theta[:, 0] < bins_dim1[j + 1])
                    & (y_col != -1)
                    & (theta[:, 1] >= bins_dim2[k])
                    & (theta[:, 1] < bins_dim2[k + 1])
                )

                if bin_mask.sum() > 0:  # bin not empty
                    y_empirical = y_col[bin_mask].mean()

                    theta_mid = torch.tensor(
                        [
                            (bins_dim1[j] + bins_dim1[j + 1]) / 2,
                            (bins_dim2[k] + bins_dim2[k + 1]) / 2,
                        ],
                        dtype=torch.float32,
                    )
                    y_theoretical = (
                        1
                        / (
                            1
                            + torch.exp(-(torch.matmul(theta_mid, single_a) + single_z))
                        ).item()
                    )

                    diff = 1 - abs(y_empirical - y_theoretical)
                    diff_list.append(diff)

    diff_array = np.array(diff_list)
    mean_diff = np.mean(diff_array)
    return mean_diff, diff_array


def goodness_of_fit_1PL_multi_dim_plot(
    z: torch.Tensor,
    theta: torch.Tensor,
    a: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bin_size: int = 6,
):
    mean_diff, diff_array = goodness_of_fit_1PL_multi_dim(z, theta, a, y, bin_size)

    sample_means = []
    for _ in range(100):
        indices = np.random.choice(
            len(diff_array), int(0.8 * len(diff_array)), replace=False
        )
        sample_mean = np.mean(diff_array[indices])
        sample_means.append(sample_mean)
    std_diff = np.std(sample_means)

    plt.figure(figsize=(10, 6))
    plt.hist(diff_array, bins=40, density=True, alpha=0.4)
    plt.xlabel(r"Difference between empirical and theoretical $P(y=1)$", fontsize=30)
    plt.ylabel(r"Goodness of fit", fontsize=30)
    plt.tick_params(axis="both", labelsize=25)
    plt.xlim(0, 1)
    plt.axvline(mean_diff, linestyle="--")
    plt.text(
        mean_diff,
        plt.gca().get_ylim()[1],
        f"{mean_diff:.2f} $\\pm$ {3 * std_diff:.2f}",
        ha="center",
        va="bottom",
        fontsize=25,
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return mean_diff, std_diff


def load_ctt_corr_scores(
    theta: np.array,
    y: np.array,
):
    assert y.shape[0] == theta.shape[0], f"{y.shape[0]} != {theta.shape[0]}"

    ctt_scores = []
    irt_scores = []

    for row, th in zip(y, theta):
        valid_values = row[row != -1]
        if len(valid_values) > 0:
            ctt_scores.append(valid_values.mean())
            irt_scores.append(th[0])

    ctt_scores = torch.stack(ctt_scores)
    irt_scores = torch.stack(irt_scores)

    if len(ctt_scores.unique()) <= 3:
        warnings.warn(
            f"ctt_scores has little value: {ctt_scores.tolist()}", UserWarning
        )

    return irt_scores, ctt_scores


def load_theta_corr_scores(
    data_folder: str,
    theta: np.array,
    dataset: str,
):
    y_model_names = pd.read_csv(
        f"{data_folder}/{dataset}/matrix.csv", index_col=0
    ).index.tolist()

    helm_df = pd.read_csv(
        f"{data_folder}/gather_data/crawl_real/helm_score/{dataset}.csv"
    )
    helm_models = helm_df["model_name"].tolist()
    helm_models = [HELM_MODEL_MAP[m] if m in HELM_MODEL_MAP else m for m in helm_models]
    helm_scores = helm_df["score"].values

    assert helm_scores.shape[0] == theta.shape[0]
    assert set(helm_models) == set(y_model_names)

    helm_df_aligned = pd.DataFrame({"model_name": helm_models, "score": helm_scores})
    theta_df_aligned = pd.DataFrame(
        {"model_name": y_model_names, "theta": theta.squeeze(-1).tolist()}
    )
    merged_df = pd.merge(
        helm_df_aligned, theta_df_aligned, on="model_name", how="inner"
    )

    aligned_helm_scores = merged_df["score"].values
    aligned_theta = merged_df["theta"].values

    aligned_helm_scores = torch.tensor(aligned_helm_scores).to(theta)
    aligned_theta = torch.tensor(aligned_theta).to(theta)

    return aligned_theta, aligned_helm_scores
