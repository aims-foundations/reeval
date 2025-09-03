import torch
import numpy as np
import scipy.stats as st
import pandas as pd
import os
from simpler_model import compute_r
from torchmetrics import AUROC
from torch.distributions import Bernoulli
# def summarize_distr(distr):
#     if  distr is None or len(distr) == 0:
#         return {"mean": None, "std": None, "ci95": (None, None)}
#     arr = np.array(distr, dtype=float)
#     n = len(arr)
#     mean = np.mean(arr)
#     std = np.std(arr, ddof=1)  # sample std (unbiased)
    
#     # t critical value for 95% CI
#     ci_low, ci_high = st.t.interval(
#         0.95, df=n-1, loc=mean, scale=std/np.sqrt(n)
#     )
    
#     return {"mean": mean, "std": std, "ci95": (ci_low, ci_high)}

def compute_metrics(P_hat, Y, test_idtor):
    # compute auc, corr log P
    auroc = AUROC(task="binary")
    test_auc = auroc(P_hat[test_idtor].cpu(), Y[test_idtor].cpu())
    r_test, _, _ = compute_r(P_hat.cpu(), Y.cpu(), test_idtor.cpu())
    log_p = (Bernoulli(probs=P_hat).log_prob(Y)*test_idtor).mean()
    return r_test, test_auc, log_p

def load_raw(config_name):
    raw_data_package = torch.load(f"results/raw/raw_data_package_{config_name}.pt")
    
    
    return raw_data_package
    
def get_metrics_from_raw(config_name):
    raw_data_package = load_raw(config_name)
    r_test, test_auc, log_p_test = compute_metrics(raw_data_package['P_hat'], raw_data_package['Y'], raw_data_package['test_idtor'])
    r_train, train_auc, log_p_train = compute_metrics(raw_data_package['P_hat'], raw_data_package['Y'], raw_data_package['train_idtor'])
    return {
        "train":{"auc": train_auc, "corr": r_train, "log_p":log_p_train },
        "test":{"auc": test_auc, "corr": r_test, "log_p":log_p_test }
    }
    


def summarize_distr(distr):
    """
    distr: 1D iterable of trials (may include NaNs)
    returns: dict with mean, std, ci95=(p2.5, p97.5)
    """
    arr = np.asarray(distr, dtype=float)
    # mask NaNs for stats that aren't nan-aware
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return {"mean": None, "std": None, "ci95": (None, None)}

    mean = arr[valid].mean()
    std  = arr[valid].std(ddof=1) if valid.sum() > 1 else 0.0

    if len(valid) == 1:
        return {"mean": mean, "std": None, "ci95": (None, None)}
    
    # empirical 2.5% / 97.5% (use nanquantile for convenience)
    p2p5, p97p5 = np.nanquantile(arr, [0.025, 0.975], method="linear")
    return {"mean": mean, "std": std, "ci95": (p2p5, p97p5)}

    # mask_list = ["random_mask","random_row","date","size"]
    # dataset_list = ["official_provider","HELM"]
    # factor_list = [1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]


metrics = ['auc', 'corr', 'log_p']
# datasets = ["HELM","official_provider","everything"]
datasets = ["HELM","official_provider",'everything']
masking_methods = ["random_mask","random_row","date","size"]
# factors = [i for i in range(1,51)]
factors = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
potential_trial = [i for i in range(100)]
# potential_trial = [i for i in range(100)]
data_list = []

# for metric in metrics:
for dataset in datasets:
    for masking_method in masking_methods:
        print(f"loading {dataset} {masking_method}")
        for K_fit in factors:
            # if dataset == 'everything' and K_fit not in [0,4]:
            #     # breakpoint()
            #     continue
            try:
                config_name = f"{dataset}_{masking_method}_k{K_fit}_i0"
                performance = get_metrics_from_raw(config_name)
            except:
                print(f"{config_name} not ready yet")
                r_test, test_auc, log_p  = None, None, None

            for metric in metrics:
                train_distr = [performance["train"][metric]]
                test_distr = [performance["test"][metric]]

                run_data = {
                    "metric":metric,
                    "dataset":dataset,
                    "masking_method":masking_method,
                    "K_fit":K_fit,
                    "distr_summary":{
                        "test_dist":summarize_distr(test_distr),
                        "train_dist":summarize_distr(train_distr)}
                }
                data_list.append(run_data)

        
flat_records = []
for entry in data_list:
    for split in ["train_dist", "test_dist"]:
        summary = entry["distr_summary"][split]
        flat_records.append({
            "metric": entry["metric"],
            "dataset": entry["dataset"],
            "masking_method": entry["masking_method"],
            "K_fit": entry["K_fit"],
            "split": split,  # train/test
            "mean": summary["mean"],
            "std": summary["std"],
            "ci95_low": summary["ci95"][0],
            "ci95_high": summary["ci95"][1],
        })
   
df = pd.DataFrame(flat_records)

# if you want to set a useful index:
# df.set_index(["metric", "dataset", "masking_method", "K_fit", "split"], inplace=True)

# print(df.head())

# pd.DataFrame(flat_records).to_csv("results/summary/partial_results_auc_corr.csv", index=False)
# breakpoint()
# df_2 = pd.read_csv("results/summary/partial_results_auc_corr_everything.csv")
# df_combined = pd.concat([df, df_2], ignore_index=True)

df.to_csv("results/summary/partial_results_auc_corr_all_2.csv", index=False)

