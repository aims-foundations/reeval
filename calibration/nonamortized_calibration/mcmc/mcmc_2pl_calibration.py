import argparse
import os

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import torch
import wandb
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm
from tueplots import bundles
from utils import (
    item_response_fn_2PL,
    item_response_fn_2PL_jnp,
    set_seed,
    theta_corr_ctt_plot,
)

plt.rcParams.update(bundles.icml2022())
plt.style.use("seaborn-v0_8-paper")


def model(question_num, testtaker_num, response_matrix):
    z2_hat = numpyro.sample("z2_hat", dist.LogNormal(0.0, 0.5).expand((question_num,)))
    z3_hat = numpyro.sample("z3_hat", dist.Normal(0.0, 1.0).expand((question_num,)))
    theta_hat = numpyro.sample(
        "theta_hat", dist.Normal(0.0, 1.0).expand((testtaker_num,))
    )

    z2_hat_expanded = jnp.expand_dims(z2_hat, 0)  # Shape: (1, question_num)
    z3_hat_expanded = jnp.expand_dims(z3_hat, 0)  # Shape: (1, question_num)
    theta_hat_expanded = jnp.expand_dims(theta_hat, 1)  # Shape: (testtaker_num, 1)
    prob_matrix = item_response_fn_2PL_jnp(
        z2_hat_expanded,
        z3_hat_expanded,
        theta_hat_expanded,
    )
    mask = response_matrix != -1
    numpyro.sample("obs", dist.Bernoulli(prob_matrix[mask]), obs=response_matrix[mask])
    # numpyro.sample("obs", dist.Bernoulli(prob_matrix), obs=response_matrix)


def irt_mcmc(
    question_num,
    testtaker_num,
    response_matrix,
    num_samples=9000,
    num_warmup=1000,
    key=0,
):
    rng_key = random.PRNGKey(key)
    rng_key, rng_key_ = random.split(rng_key)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        question_num=question_num,
        testtaker_num=testtaker_num,
        response_matrix=response_matrix,
    )
    # mcmc.print_summary()

    theta_samples = mcmc.get_samples()["theta_hat"]
    z2_samples = mcmc.get_samples()["z2_hat"]
    z3_samples = mcmc.get_samples()["z3_hat"]
    return theta_samples, z2_samples, z3_samples


def goodness_of_fit_2PL(
    theta: torch.Tensor,
    z2_samples: torch.Tensor,
    z3_samples: torch.Tensor,
    y: torch.Tensor,
    bin_size: int = 6,
):
    bin_start, bin_end = torch.min(theta), torch.max(theta)
    bins = torch.linspace(bin_start, bin_end, bin_size + 1)
    print(bins)

    diff_list = []
    for i in tqdm(range(y.shape[1])):
        single_z2_samples = z2_samples[:, i]
        single_z3_samples = z3_samples[:, i]
        y_col = y[:, i]

        for j in range(len(bins) - 1):
            bin_mask = (theta >= bins[j]) & (theta < bins[j + 1]) & (y_col != -1)
            if bin_mask.sum() > 0:  # Bin not empty
                y_empirical = y_col[bin_mask].mean()

                theta_mid = (bins[j] + bins[j + 1]) / 2
                y_theoretical = item_response_fn_2PL(
                    single_z2_samples, single_z3_samples, theta_mid
                )
                in_diff_list = [1 - abs(y_empirical - yt) for yt in y_theoretical]
                diff = sum(in_diff_list) / len(in_diff_list)
                diff_list.append(diff)

    diff_array = np.array(diff_list)
    mean_diff = diff_array.mean()
    return mean_diff, diff_array


def goodness_of_fit_2PL_plot(
    theta: torch.Tensor,
    z2_samples: torch.Tensor,
    z3_samples: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bin_size: int = 6,
):
    mean_diff, diff_array = goodness_of_fit_2PL(
        theta, z2_samples, z3_samples, y, bin_size
    )
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


if __name__ == "__main__":
    # wandb.init(project="em_2pl_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    y = pd.read_csv(
        f"../data/pre_calibration/{args.dataset}/matrix.csv", index_col=0
    ).values
    testtaker_num, question_num = y.shape

    output_dir = f"../data/em_2pl_calibration/{args.dataset}"
    plot_dir = f"../plot/em_2pl_calibration"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    theta_path = f"{output_dir}/theta.csv"
    z2_path = f"{output_dir}/z2.csv"
    z3_path = f"{output_dir}/z3.csv"

    theta_samples_path = f"{output_dir}/theta_samples.npy"
    z2_samples_path = f"{output_dir}/z2_samples.npy"
    z3_samples_path = f"{output_dir}/z3_samples.npy"

    theta_samples, z2_samples, z3_samples = irt_mcmc(question_num, testtaker_num, y)
    theta_samples = np.array(theta_samples)  # (num_samples, testtaker_num)
    z2_samples = np.array(z2_samples)
    z3_samples = np.array(z3_samples)

    np.save(theta_samples_path, theta_samples)
    np.save(z2_samples_path, z2_samples)
    np.save(z3_samples_path, z3_samples)

    theta_hat = theta_samples.mean(axis=0)
    theta_df = pd.DataFrame({"theta": theta_hat})
    z2_df = pd.DataFrame({"z2": z2_samples.mean(axis=0)})
    z3_df = pd.DataFrame({"z3": z3_samples.mean(axis=0)})

    theta_df.to_csv(theta_path, index=False)
    z2_df.to_csv(z2_path, index=False)
    z3_df.to_csv(z3_path, index=False)

    # theta_hat = pd.read_csv(theta_path)['theta'].values
    # z2_samples = np.load(z2_samples_path)
    # z3_samples = np.load(z3_samples_path)

    _, _ = goodness_of_fit_2PL_plot(
        theta=torch.tensor(theta_hat, dtype=torch.float32),
        z2_samples=torch.tensor(z2_samples, dtype=torch.float32),
        z3_samples=torch.tensor(z3_samples, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        plot_path=f"{plot_dir}/goodness_of_fit_{args.dataset}",
    )

    _, _ = theta_corr_ctt_plot(
        theta=theta_hat,
        y=y,
        plot_path=f"{plot_dir}/theta_corr_ctt_{args.dataset}",
    )
