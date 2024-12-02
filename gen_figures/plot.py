import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics import SpearmanCorrCoef
from tueplots import bundles
from utils.utils import goodness_of_fit

plt.rcParams.update(bundles.iclr2024())
plt.style.use("seaborn-v0_8-paper")

from amortized_irt import IRT
from utils.constants import PLOT_NAME_MAP


def theta_corr_plot(
    mode: str,
    theta: np.array,
    plot_path: str,
    ctt_score: np.array = None,
    helm_score: np.array = None,
):
    if theta.shape[-1] > 1:
        print("Theta correlation is only supported for 1D.")
        return np.nan, np.nan

    sm_fn = SpearmanCorrCoef()
    if mode == "ctt":
        external_scores = ctt_score
        #### PATCHING SOLUTION FOR BAD DATA ####
        nan_mask = ~torch.isnan(external_scores)
        theta = theta[nan_mask]
        external_scores = external_scores[nan_mask]
        #######################################
    elif mode == "helm":
        # Check all external scores are nan
        nan_helm_idxs = torch.isnan(helm_score)
        if nan_helm_idxs.all():
            return np.nan, np.nan

        theta = theta[~nan_helm_idxs]
        external_scores = helm_score[~nan_helm_idxs]

    corr = sm_fn(theta, external_scores).item()

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


def auc_roc_plot(
    item_parms: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bootstrap_size: int = 100,
):
    n_items = item_parms.shape[0]
    n_testtakers = theta.shape[0]

    assert y.shape[0] == n_testtakers, f"{y.shape[0]} != {n_testtakers}"
    assert y.shape[1] == n_items, f"{y.shape[1]} != {n_items}"

    item_parms_list = [*(item_parms[..., 0:3].T), item_parms[..., 3:]]
    probs_theoretical = IRT.compute_prob(theta, *item_parms_list)
    y_mask = y != -1
    y = y[y_mask]
    probs_theoretical = probs_theoretical[y_mask]
    auc_roc = roc_auc_score(y.cpu(), probs_theoretical.cpu())

    sample_accs = []
    num_dp = len(y)
    for _ in range(bootstrap_size):
        indices = np.random.choice(num_dp, int(0.8 * num_dp), replace=False)
        sample_acc = roc_auc_score(y[indices].cpu(), probs_theoretical[indices].cpu())
        sample_accs.append(sample_acc)
    std_auc_roc = np.std(sample_accs)

    return auc_roc, std_auc_roc


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
