import random
import warnings
from typing import List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from embed_text_package.embed_text import Embedder
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torchmetrics import SpearmanCorrCoef
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
plt.style.use("seaborn-v0_8-paper")
import seaborn as sns

from .constants import HELM_MODEL_MAP, PLOT_NAME_MAP
from .irt import IRT


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


def get_embed(
    dataset,
    cols_to_be_embded=["text"],
    bs=1024,
    model_name="meta-llama/Meta-Llama-3-8B",
):
    embdr = Embedder()
    embdr.load(model_name)
    dataloader = DataLoader(dataset, batch_size=bs)
    emb = embdr.get_embeddings(dataloader, model_name, cols_to_be_embded)
    return emb["text"]


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


def goodness_of_fit_plot(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bin_size: int = 6,
):
    diff_array = goodness_of_fit(z, theta, y, bin_size)
    mean_diff = np.mean(diff_array)

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
        f"{data_folder}/pre_calibration/{dataset}/matrix.csv", index_col=0
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


def theta_corr_plot(
    mode: str,
    theta: np.array,
    plot_path: str,
    data_folder: str = None,
    y: np.array = None,
    dataset: np.array = None,
):
    if dataset == "airbench" and mode == "helm":
        print("airbench dataset does not have HELM scores.")
        return None, None

    if theta.shape[-1] > 1:
        print("Theta correlation is only supported for 1D.")
        return None, None

    sm_fn = SpearmanCorrCoef()
    if mode == "ctt":
        theta, external_scores = load_ctt_corr_scores(theta, y)
    elif mode == "helm":
        theta, external_scores = load_theta_corr_scores(data_folder, theta, dataset)
    corr = sm_fn(theta, external_scores)

    sample_corrs = []
    n_bootstrap_samples = 100

    for _ in range(n_bootstrap_samples):
        indices = np.random.choice(len(theta), int(0.8 * len(theta)), replace=False)

        sample_corr = sm_fn(theta[indices], external_scores[indices])
        sample_corrs.append(sample_corr)
    sample_std = torch.std(torch.stack(sample_corrs)).item()

    plt.figure(figsize=(18, 10))
    plt.scatter(theta.cpu(), external_scores.cpu())
    plt.xlabel(r"$\theta$ from calibration", fontsize=45)
    plt.ylabel(f"{mode.upper()} score", fontsize=45)
    plt.title(f"Correlation: {corr:.2f} $\\pm$ {3 * sample_std:.2f}", fontsize=45)
    plt.tick_params(axis="both", labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return corr, sample_std


def accuracy_plot(
    item_parms: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
):
    n_items = item_parms.shape[0]
    n_testtakers = theta.shape[0]

    assert y.shape[0] == n_testtakers, f"{y.shape[0]} != {n_testtakers}"
    assert y.shape[1] == n_items, f"{y.shape[1]} != {n_items}"

    item_parms_list = [*(item_parms[..., 0:3].T), item_parms[..., 3:]]
    probs_theoretical = IRT.compute_prob(theta, *item_parms_list)
    y_theoretical = (probs_theoretical > 0.5).float()

    accuracy = (y_theoretical == y).float().mean(dim=0)

    sample_accs = []
    for _ in range(100):
        indices = np.random.choice(
            len(accuracy), int(0.8 * len(accuracy)), replace=False
        )
        sample_acc = accuracy[indices].mean()
        sample_accs.append(sample_acc)
    std_acc = torch.std(torch.stack(sample_accs)).item()

    plt.figure(figsize=(10, 6))
    plt.hist(accuracy.cpu(), bins=40, density=True, alpha=0.4)
    plt.xlabel(r"Accuracy", fontsize=30)
    plt.ylabel(r"Frequency", fontsize=30)
    plt.tick_params(axis="both", labelsize=25)
    plt.xlim(0, 1)
    plt.axvline(accuracy.mean().item(), linestyle="--")
    plt.text(
        accuracy.mean().item(),
        plt.gca().get_ylim()[1],
        f"{accuracy.mean().item():.2f} $\\pm$ {3 * std_acc:.2f}",
        ha="center",
        va="bottom",
        fontsize=25,
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return accuracy.mean(), std_acc


def error_bar_plot_single(datasets, means, stds, plot_path, xlabel, xlim_upper=1.1):
    datasets = [PLOT_NAME_MAP[dataset] for dataset in datasets]
    sorted_data = sorted(zip(datasets, means, stds), key=lambda x: x[1])
    datasets, means, stds = zip(*sorted_data)
    stds_mul3 = [s * 3 for s in stds]

    fig, ax = plt.subplots(figsize=(8, 18))
    ax.barh(
        datasets,
        means,
        xerr=[np.zeros(len(datasets)), stds_mul3],
        capsize=5,
        color="blue",
        alpha=0.4,
        error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "blue"},
    )

    ax.set_xlabel(xlabel, fontsize=35)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlim(0, xlim_upper)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def error_bar_plot_double(
    datasets,
    means_train,
    stds_train,
    means_test,
    stds_test,
    plot_path,
    xlabel,
    xlim_upper=1.1,
    plot_std=True,
    average_line=False,
):
    datasets = [PLOT_NAME_MAP[dataset] for dataset in datasets]
    sorted_data = sorted(
        zip(datasets, means_train, stds_train, means_test, stds_test),
        key=lambda x: x[3],
    )
    datasets, means_train, stds_train, means_test, stds_test = zip(*sorted_data)
    fig, ax = plt.subplots(figsize=(8, 18))

    if plot_std:
        stds_train_mul3 = [s * 3 for s in stds_train]
        stds_test_mul3 = [s * 3 for s in stds_test]
        ax.barh(
            datasets,
            means_train,
            xerr=[np.zeros(len(datasets)), stds_train_mul3],
            capsize=5,
            color="blue",
            alpha=0.4,
            error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "blue"},
        )
        ax.barh(
            datasets,
            means_test,
            xerr=[np.zeros(len(datasets)), stds_test_mul3],
            capsize=5,
            color="orange",
            alpha=0.4,
            error_kw={"elinewidth": 2, "capthick": 2, "ecolor": "orange"},
        )
    else:
        ax.barh(datasets, means_train, color="blue", alpha=0.4)
        ax.barh(datasets, means_test, color="orange", alpha=0.4)
        print("")
        print(xlabel)
        improvements = []
        for dataset, mse_train, mse_test in zip(datasets, means_train, means_test):
            improvement = (mse_train - mse_test) / mse_train
            improvements.append((dataset, improvement))

        improvements.sort(key=lambda x: x[1], reverse=True)
        # print mean improvement
        print(
            f"Mean improvement: {np.mean([improvement for _, improvement in improvements])}"
        )
        for dataset, improvement in improvements:
            print(f"{dataset}: {improvement}")

    if average_line:
        avg_train = np.mean(means_train)
        avg_test = np.mean(means_test)
        ax.axvline(avg_train, color="blue", linestyle="--", linewidth=2)
        ax.axvline(avg_test, color="orange", linestyle="--", linewidth=2)
        max_y = len(datasets) - 1
        ax.text(
            avg_train - 0.5,
            max_y,
            f"{avg_train:.2f}",
            color="blue",
            fontsize=25,
            ha="center",
        )
        ax.text(
            avg_test + 0.5,
            max_y,
            f"{avg_test:.2f}",
            color="orange",
            fontsize=25,
            ha="center",
        )

    ax.set_xlabel(xlabel, fontsize=35)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlim(0, xlim_upper)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def amorz_corr_nonamorz(
    z_amor: np.array,
    z_nonamor: np.array,
):
    assert z_amor.shape == z_nonamor.shape, f"{z_amor.shape} != {z_nonamor.shape}"
    z_corr = np.corrcoef(z_amor, z_nonamor)[0, 1]
    return z_corr


def plot_corr(
    data1,
    data2,
    plot_path,
    title,
    xlabel,
    ylabel,
):
    # corr = np.corrcoef(data1, data2)[0, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(data1, data2, color="blue")
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.title(
        # title.format(corr),
        title,
        fontsize=25,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis="both", labelsize=16)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_corr_double(
    data1_train,
    data1_test,
    data2_train,
    data2_test,
    plot_path,
    xlabel,
    ylabel,
):
    # corr_train = np.corrcoef(data1_train, data2_train)[0, 1]
    # corr_test = np.corrcoef(data1_test, data2_test)[0, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(data1_train, data2_train, color="blue", label="Train")
    plt.scatter(data1_test, data2_test, color="red", label="Test")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        r"Goodness of Fit",
        # r'Goodness of Fit. $\rho_\mathrm{{train}}$ = {:.2f}, $\rho_\mathrm{{test}}$ = {:.2f}'.format(corr_train, corr_test),
        fontsize=25,
    )
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(fontsize=16)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar(datasets, nums, plot_path, ylabel, exp_axis=False):
    datasets = [PLOT_NAME_MAP[dataset] for dataset in datasets]
    sorted_by_nums = sorted(zip(datasets, nums), key=lambda x: x[1])
    sorted_datasets, sorted_nums = zip(*sorted_by_nums)
    plt.figure(figsize=(25, 10))
    bars = plt.bar(sorted_datasets, sorted_nums)
    plt.xticks(rotation=30, ha="right", fontsize=35)
    plt.tick_params(axis="both", labelsize=35)
    plt.ylabel(ylabel, fontsize=35)
    for bar, num in zip(bars, sorted_nums):
        height = bar.get_height()
        if height >= 1000:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height/1000:.1f}k",
                ha="center",
                va="bottom",
                fontsize=20,
            )
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height}",
                ha="center",
                va="bottom",
                fontsize=20,
            )
    if exp_axis:
        plt.yscale("log")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hist(
    data,
    plot_path,
    ylabel,
):
    plt.figure(figsize=(6, 6))
    plt.hist(data, bins=30, density=True, alpha=0.4)
    mean_value = np.mean(data)
    plt.axvline(mean_value, linestyle="--", linewidth=2)
    plt.text(
        mean_value,
        plt.gca().get_ylim()[1],
        f"{mean_value:.2f}",
        fontsize=16,
        ha="center",
    )
    plt.ylabel(ylabel, fontsize=25)
    plt.tick_params(axis="both", labelsize=16)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cat(
    randoms,
    cats,
    cat_subs,
    plot_path,
    ylabel,
):
    plt.figure(figsize=(6, 6))
    plt.plot(randoms, label="Random", color="red", linewidth=2)
    plt.plot(cats, label="Fisher large", color="blue", linewidth=2)
    plt.plot(cat_subs, label="Fisher small", color="darkgoldenrod", linewidth=2)
    plt.tick_params(axis="both", labelsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
