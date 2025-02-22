import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import spearman_corrcoef
import warnings
warnings.filterwarnings("ignore")

scenarios = [
    "mmlu",
    # "babi_qa",
    # "bbq",
    # "blimp",
    # "boolq",
    # "civil_comments",
    # "commonsense",
    # "dyck_language_np=3",
    # "entity_data_imputation",
    # "entity_matching",
    # "gsm",
    # "imdb",
    # "legal_support",
    # "lsat_qa",
    # "synthetic_reasoning",
    # "truthful_qa",
    # "wikifact"
]

def gof(
    item_parms: torch.Tensor,  # n_items x n_PL
    theta: torch.Tensor,  # n_testtakers x n_D
    y: torch.Tensor, # n_testtakers x n_items
    n_bins: int = 6,
):
    n_items = item_parms.shape[0]
    n_testtakers = theta.shape[0]
    n_D = theta.shape[1]
    
    assert y.shape[0] == n_testtakers, f"{y.shape[0]} != {n_testtakers}"
    assert y.shape[1] == n_items, f"{y.shape[1]} != {n_items}"

    bins_start = torch.min(theta, dim=0).values # n_D
    bins_end = torch.max(theta, dim=0).values # n_D
    bins = torch.stack(
        [
            torch.linspace(bin_start, bin_end, n_bins + 1).to(theta)
            for bin_start, bin_end in zip(bins_start, bins_end)
        ],
        dim=-1,
    )
    # >>> n_bins+1 x n_D
    
    thetas_mid = (bins[:-1] + bins[1:]) / 2 # n_bins x n_D
    # TODO: modify for multiple PL
    probs_theoretical = torch.sigmoid(thetas_mid + item_parms.T) # n_bins x n_items
    bin_masks = (theta[:, None] >= bins[:-1]) & (theta[:, None] < bins[1:]) # n_testtakers x n_bins x n_D

    diffs = []
    for d in range(n_D):
        diff_D = []
        for bi in range(n_bins):
            bin_mask = bin_masks[:, bi, d] # n_testtakers
            if bin_mask.sum() <= 0:
                continue

            y_bins = y[bin_mask]
            prob_empirical = y_bins.nanmean(dim=0) # n_items

            nan_mask = torch.isnan(prob_empirical)
            prob_theoretical = probs_theoretical[bi] # n_items

            diff = 1 - torch.abs(prob_empirical - prob_theoretical)[~nan_mask]
            diff_D.extend(diff.tolist())

        diffs.append(np.array(diff_D))
    
    diff_array = np.concatenate(diffs)
    mean_diff = np.mean(diff_array)
    sample_means = []
    for _ in range(100):
        indices = np.random.choice(
            len(diff_array), int(0.8 * len(diff_array)), replace=False
        )
        sample_mean = np.mean(diff_array[indices])
        sample_means.append(sample_mean)
    std_diff = np.std(sample_means)
    
    return mean_diff, std_diff


def auc_roc(
    item_parms: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    bootstrap_size: int = 100,
):
    n_items = item_parms.shape[0]
    n_testtakers = theta.shape[0]

    assert y.shape[0] == n_testtakers, f"{y.shape[0]} != {n_testtakers}"
    assert y.shape[1] == n_items, f"{y.shape[1]} != {n_items}"

    probs_theoretical = torch.sigmoid(theta + item_parms.T)
    nan_mask = torch.isnan(y)
    y = y[~nan_mask]
    probs_theoretical = probs_theoretical[~nan_mask]
    auc_roc = roc_auc_score(y.detach().numpy(), probs_theoretical.detach().numpy())

    sample_accs = []
    num_dp = len(y)
    for _ in range(bootstrap_size):
        indices = np.random.choice(num_dp, int(0.8 * num_dp), replace=False)
        sample_acc = roc_auc_score(y[indices].detach().numpy(), probs_theoretical[indices].detach().numpy())
        sample_accs.append(sample_acc)
    std_auc_roc = np.std(sample_accs)

    return auc_roc, std_auc_roc

if __name__ == "__main__":
    torch.manual_seed(0)
    
    res_file = open("../data/trad_calibrate/trad_calibrate_result.txt", "w")
    for scenario in scenarios:
        print(f"\nProcessing {scenario}")
        res_file.write(f"\nProcessing {scenario}")
        
        save_dir = f"../data/trad_calibrate/{scenario}"
        matrix = torch.load(f"{save_dir}/matrix.pt")
        train_mask = torch.load(f"{save_dir}/train_mask.pt")
        z = torch.load(f"{save_dir}/z.pt")
        theta = torch.load(f"{save_dir}/theta.pt")
        
        train_matrix = matrix.clone()
        train_matrix[~train_mask] = torch.nan
        test_matrix = matrix.clone()
        test_matrix[train_mask] = torch.nan
        
        print(f"z spearman with CTT: {spearman_corrcoef(z, matrix.nanmean(0)):.4f}")
        res_file.write(f"z spearman with CTT: {spearman_corrcoef(z, matrix.nanmean(0)):.4f}\n")
        
        print(f"theta spearman with CTT: {spearman_corrcoef(theta, matrix.nanmean(1)):.4f}")
        res_file.write(f"theta spearman with CTT: {spearman_corrcoef(theta, matrix.nanmean(1)):.4f}\n")
        
        train_gof_mean, train_gof_std = gof(z[:, None], theta[:, None], train_matrix)
        test_gof_mean, test_gof_std = gof(z[:, None], theta[:, None], test_matrix)
        print(f"Train GOF: {train_gof_mean:.4f} ± {train_gof_std:.4f}")
        res_file.write(f"Train GOF: {train_gof_mean:.4f} ± {train_gof_std:.4f}\n")
        print(f"Test GOF: {test_gof_mean:.4f} ± {test_gof_std:.4f}")
        res_file.write(f"Test GOF: {test_gof_mean:.4f} ± {test_gof_std:.4f}\n")
        
        # TODO: to speed up, temporarily using bootstrap_size=5
        train_auc, train_auc_std = auc_roc(z[:, None], theta[:, None], train_matrix, 5)
        test_auc, test_auc_std = auc_roc(z[:, None], theta[:, None], test_matrix, 5)
        print(f"Train AUC-ROC: {train_auc:.4f} ± {train_auc_std:.4f}")
        res_file.write(f"Train AUC-ROC: {train_auc:.4f} ± {train_auc_std:.4f}\n")
        print(f"Test AUC-ROC: {test_auc:.4f} ± {test_auc_std:.4f}")
        res_file.write(f"Test AUC-ROC: {test_auc:.4f} ± {test_auc_std:.4f}\n")
        
        res_file.flush()
        