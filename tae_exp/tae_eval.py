import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from calibration.nonamortized_calibration.calibrate import calibrate
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.irt import IRT
from utils.utils import inverse_sigmoid, set_seed, str2bool


def plot_hard_easy(
    theta_hats: list,
    y_means: list,
    theta: float,
    y_mean: float,
    plot_path: str,
):
    plt.figure(figsize=(8, 6))
    plt.hist(
        theta_hats,
        bins=40,
        color="red",
        alpha=0.2,
        label="IRT Estimation",
        density=True,
    )
    plt.hist(
        y_means, bins=40, color="blue", alpha=0.2, label="CTT Estimation", density=True
    )
    plt.axvline(x=theta, color="red", linestyle="-", linewidth=2)
    plt.axvline(x=y_mean, color="blue", linewidth=2)
    sns.kdeplot(theta_hats, color="red", linewidth=2, bw_adjust=2)
    plt.xlabel(r"Ability", fontsize=25)
    plt.xlim(-6, 6)
    plt.ylabel(r"Density", fontsize=25)
    plt.legend(fontsize=20)
    plt.tick_params(axis="both", labelsize=20)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_mle(y, z, max_epoch=1000):
    ability = torch.zeros((1, 1), device=y.device, requires_grad=True)

    optimizer = torch.optim.Adam([ability], lr=0.01)
    pbar = tqdm(range(max_epoch))

    for _ in pbar:
        prob_matrix = IRT.compute_prob(
            ability=ability,
            difficulty=z,
            disciminatory=1,
            guessing=0,
            loading_factor=1,
        )

        mask = y != -1
        masked_response_matrix = y[mask]
        masked_prob_matrix = prob_matrix[mask]

        berns = torch.distributions.Bernoulli(probs=masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix({"loss": loss.item()})

    return ability.detach()


def bootstrap_z(z, single_y, subset_size, n_bootstrap=10):
    n_questions = z.size(0)

    auc_ctts = []
    auc_irts = []
    auc_inv_sigmoids = []

    for _ in range(n_bootstrap):
        # Step 2: Sample 2 subset of questions (and corresponding z) with size = 100
        subset1 = torch.randperm(n_questions)[:subset_size]
        subset2 = torch.randperm(n_questions)[:subset_size]

        # Step 3: On the first subset:
        #      - Compute CTT scores for X via averaging all scores on this subset
        #      - Compute IRT scores (theta) for X via MLE
        avg_ctt_score = single_y[subset1].mean()
        theta = run_mle(single_y[subset1].unsqueeze(0), z[subset1])

        # Step 4: On the second subset: Compute AUC-ROC for binary classifier whose
        #      - probability is CTT score, applying uniformly for all questions (disregard z)
        #      - probability is IRT score + difficulty
        #      - probability is inverse sigmoid of CTT score + difficulty
        ground_truth = single_y[subset2]
        z_subset2 = z[subset2]

        # Compute CTT scores
        ctt_probs = torch.bernoulli(avg_ctt_score.unsqueeze(0).expand(subset_size))
        auc_ctt = roc_auc_score(ground_truth.cpu().numpy(), ctt_probs.cpu().numpy())

        # Compute IRT scores
        irt_probs = IRT.compute_prob(
            ability=theta,
            difficulty=z_subset2,
            disciminatory=1,
            guessing=0,
            loading_factor=1,
        )
        irt_probs = irt_probs.squeeze(0)
        auc_irt = roc_auc_score(ground_truth.cpu().numpy(), irt_probs.cpu().numpy())

        # Compute inverse sigmoid scores
        inv_sigmoid_theta = inverse_sigmoid(avg_ctt_score)
        inv_sigmoid_probs = IRT.compute_prob(
            ability=inv_sigmoid_theta.unsqueeze(0).reshape(1, 1),
            difficulty=z_subset2,
            disciminatory=1,
            guessing=0,
            loading_factor=1,
        )
        inv_sigmoid_probs = inv_sigmoid_probs.squeeze(0)
        auc_inv_sigmoid = roc_auc_score(
            ground_truth.cpu().numpy(), inv_sigmoid_probs.cpu().numpy()
        )

        # Save results
        auc_ctts.append(auc_ctt)
        auc_irts.append(auc_irt)
        auc_inv_sigmoids.append(auc_inv_sigmoid)

    return auc_ctts, auc_irts, auc_inv_sigmoids


def bootstrap_X(args, response_matrix, subset_size=100, n_bootstrap=10, device="cpu"):
    n_models, n_questions = response_matrix.shape

    # Randomly select n_bootstrap test takers without replacement
    test_takers = torch.randperm(n_models)[:n_bootstrap]

    results = []
    for tt_idx in test_takers:
        # Step 1: Calibration on all questions and n-1 test takers, leave one test taker X out
        y_train = response_matrix[torch.arange(n_models) != tt_idx]
        irt_model = calibrate(
            response_matrix=y_train,
            D=args.D,
            PL=args.PL,
            fitting_method=args.fitting_method,
            max_epoch=args.max_epoch,
            device=device,
        )
        item_parms = irt_model.get_item_parameters().detach()
        z = item_parms[:, 0]

        single_y = response_matrix[tt_idx]
        auc_ctts, auc_irts, auc_inv_sigmoids = bootstrap_z(
            z, single_y, subset_size, n_bootstrap=n_bootstrap
        )

        results.append(
            {
                "tt_idx": tt_idx.item(),
                "auc_ctt": np.mean(auc_ctts),
                "auc_irt": np.mean(auc_irts),
                "auc_inv_sigmoid": np.mean(auc_inv_sigmoids),
            }
        )
        print(results[-1])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"]
    )
    parser.add_argument("--max_epoch", type=int, default=3000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selection_prob = 0.8
    subset_size = 100
    step_size = 4000
    iterations = 1024
    batch_size = 1024
    assert batch_size % 2 == 0, "Batch size should divide by 2"

    y = pd.read_csv(
        f"../data/pre_calibration/{args.dataset}/matrix.csv", index_col=0
    ).values
    y = torch.tensor(y, device=device).float()

    # rows in y with more than 500 non -1 values
    valid_rows_mask = (y != -1).sum(axis=1) > 500
    y = y[valid_rows_mask]

    # Step 1: Calibration on all questions and n-1 test takers, leave one test taker X out
    # Step 2: Sample 2 subset of questions (and corresponding z) with size = 100
    # Step 3: On the first subset:
    #      - Compute CTT scores for X via averaging all scores on this subset
    #      - Compute IRT scores (theta) for X via MLE
    # Step 4: On the second subset: Compute AUC-ROC for binary classifier whose
    #      - probability is CTT score, applying uniformly for all questions (disregard z)
    #      - probability is IRT score + difficulty
    #      - probability is inverse sigmoid of CTT score + difficulty
    # Step 5: Run bootstrap with different pair of subsets and different X

    results = bootstrap_X(
        args, y, subset_size=subset_size, n_bootstrap=10, device=device
    )

    # Save results
    os.makedirs(f"../data/bootstrap/{args.dataset}", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"../data/bootstrap/{args.dataset}/results.csv", index=False)
