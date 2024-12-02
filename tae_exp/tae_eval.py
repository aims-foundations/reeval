import argparse
import io
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from amortized_irt import IRT
from huggingface_hub import HfApi, snapshot_download
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.constants import DATASETS
from utils.utils import inverse_sigmoid, set_seed, str2bool

warnings.filterwarnings("ignore")


def run_mle(y, z, max_epoch=200):
    ability = torch.zeros((1, 1), device=y.device, requires_grad=True)

    optimizer = torch.optim.Adam([ability], lr=0.01)
    # pbar = tqdm(range(max_epoch))
    pbar = range(max_epoch)

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
        # pbar.set_postfix({"loss": loss.item()})

    return ability.detach()


def bootstrap_z(z, single_y, subset_size, n_question_bootstrap):
    single_y_valid = single_y[single_y != -1]
    z_valid = z[single_y != -1]
    n_questions = z_valid.size(0)

    auc_ctts = []
    auc_irts = []
    auc_inv_sigmoids = []

    score_irt = []
    score_ctt = []
    score_inv_sigmoid = []

    mean_difficulty_subset1 = []

    for _ in range(n_question_bootstrap):
        # Step 2: Sample 2 subset of questions (and corresponding z) with size = 100
        subset1 = torch.randperm(n_questions)[:subset_size]
        subset2 = torch.randperm(n_questions)[:subset_size]

        ground_truth = single_y_valid[subset2]
        # if grountruth is all 0 or 1, try again
        for _ in range(10):
            if ground_truth.sum() == 0 or ground_truth.sum() == len(ground_truth):
                subset2 = torch.randperm(n_questions)[:subset_size]
                ground_truth = single_y_valid[subset2]
            else:
                break

        # if grountruth is all 0 or 1, skip this iteration
        if ground_truth.sum() == 0 or ground_truth.sum() == len(ground_truth):
            continue

        # Step 3: On the first subset:
        #      - Compute CTT scores for X via averaging all scores on this subset
        #      - Compute IRT scores (theta) for X via MLE
        mean_difficulty_subset1.append(z_valid[subset1].mean().item())

        avg_ctt_score = single_y_valid[subset1].mean()
        score_ctt.append(avg_ctt_score.item())

        theta = run_mle(single_y_valid[subset1].unsqueeze(0), z_valid[subset1])
        score_irt.append(theta.item())

        # Step 4: On the second subset: Compute AUC-ROC for binary classifier whose
        #      - probability is CTT score, applying uniformly for all questions (disregard z)
        #      - probability is IRT score + difficulty
        #      - probability is inverse sigmoid of CTT score + difficulty
        ground_truth = single_y_valid[subset2]
        z_subset2 = z_valid[subset2]

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
        score_inv_sigmoid.append(inv_sigmoid_theta.item())
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

    # spearman corr between mean difficulty of subset 1 and irt score
    sm_irt = spearmanr(mean_difficulty_subset1, score_irt).statistic

    # spearman corr between mean difficulty of subset 1 and ctt score
    sm_ctt = spearmanr(mean_difficulty_subset1, score_ctt).statistic

    # spearman corr between mean difficulty of subset 1 and inverse sigmoid of ctt score
    sm_inv_sigmoid = spearmanr(mean_difficulty_subset1, score_inv_sigmoid).statistic

    std_irt = np.std(score_irt)
    std_inv_sigmoid = np.std(score_inv_sigmoid)

    return (
        auc_ctts,
        auc_irts,
        auc_inv_sigmoids,
        sm_irt,
        sm_ctt,
        sm_inv_sigmoid,
        std_irt,
        std_inv_sigmoid,
    )


def process_X(
    tt_idx,
    args,
    response_matrix,
    subset_size=100,
    n_question_bootstrap=100,
    device="cpu",
):
    n_models, _ = response_matrix.shape
    # Step 1: Calibration on all questions and n-1 test takers, leave one test taker X out
    y_train = response_matrix[torch.arange(n_models) != tt_idx]

    n_models, n_questions = y_train.shape
    irt_model = IRT(
        n_questions=n_questions,
        n_testtaker=n_models,
        D=args.D,
        PL=args.PL,
        amortize_item=False,
        amortize_student=False,
        amortized_question_hyperparams=None,
        amortized_model_hyperparams=None,
        device=device,
        report_to=None,
    )

    irt_model.fit(
        max_epoch=args.max_epoch,
        response_matrix=y_train,
        method=args.fitting_method,
        embedding=None,
        model_features=None,
    )

    item_parms = irt_model.get_item_parameters().detach()
    z = item_parms[:, 0]

    single_y = response_matrix[tt_idx]
    (
        auc_ctts,
        auc_irts,
        auc_inv_sigmoids,
        sm_irt,
        sm_ctt,
        sm_inv_sigmoid,
        std_irt,
        std_inv_sigmoid,
    ) = bootstrap_z(z, single_y, subset_size, n_question_bootstrap=n_question_bootstrap)

    return {
        "tt_idx": tt_idx.item(),
        "mean_auc_ctt": np.mean(auc_ctts),
        "mean_auc_irt": np.mean(auc_irts),
        "mean_auc_inv_sigmoid": np.mean(auc_inv_sigmoids),
        "std_auc_ctt": np.std(auc_ctts),
        "std_auc_irt": np.std(auc_irts),
        "std_auc_inv_sigmoid": np.std(auc_inv_sigmoids),
        "spearman_corr_irt": sm_irt,
        "spearman_corr_ctt": sm_ctt,
        "spearman_corr_inv_sigmoid": sm_inv_sigmoid,
        "std_irt": std_irt,
        "std_inv_sigmoid": std_inv_sigmoid,
    }


def bootstrap_student(
    args,
    response_matrix,
    subset_size=100,
    n_student_bootstrap=10,
    n_question_bootstrap=100,
    device="cpu",
):
    n_models, _ = response_matrix.shape

    # Randomly select n_bootstrap test takers without replacement
    test_takers = torch.randperm(n_models)[:n_student_bootstrap]

    results = []
    # Run all questions parallelly
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_X,
                tt_idx,
                args,
                response_matrix,
                subset_size,
                n_question_bootstrap,
                device,
            )
            for tt_idx in list(test_takers)
        ]
        for future in tqdm(futures, desc="Test taker"):
            results.append(future.result())
            print(results[-1])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="airbench")
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fitting_method", type=str, default="mle", choices=["mle", "mcmc", "em"]
    )
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--subset_size", type=int, default=50)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--n_student_bootstrap", type=int, default=10)
    parser.add_argument("--n_question_bootstrap", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download and load data
    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_responses", repo_type="dataset"
    )

    y = torch.load(f"{data_folder}/{args.dataset}/response_matrix.pt").to(
        device=device, dtype=torch.float32
    )

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

    results = bootstrap_student(
        args,
        y,
        subset_size=args.subset_size,
        n_student_bootstrap=args.n_student_bootstrap,
        n_question_bootstrap=args.n_question_bootstrap,
        device=device,
    )

    # Save results
    upload_api = HfApi()
    df = pd.DataFrame(results)
    results_file = io.BytesIO()
    pd.DataFrame(results).to_csv(results_file, index=False)
    upload_api.upload_file(
        repo_id="stair-lab/reeval_results",
        repo_type="dataset",
        path_in_repo=f"tae/{args.dataset}.csv",
        path_or_fileobj=results_file,
    )
