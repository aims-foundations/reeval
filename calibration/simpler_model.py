import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torchmetrics import AUROC
from util import get_helm_benchmark, get_official_provider_benchmark, get_everything_data_sk2, get_mask_and_data,get_everything_benchmark
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import sys
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import traceback


# import wandb
class LogisticMF(nn.Module):
    def __init__(self, N, M, K):
        super().__init__()
        self.U = nn.Parameter(torch.randn(N, K))
        self.V = nn.Parameter(torch.randn(M, K))

    def forward(self):
        return self.U @ self.V.T

def fit_logistic_mf(Y, K, mask=None, steps=1000, lr=1e-2, verbose=True, device=None):
    if device is None:
        device = Y.device
    Y = Y.to(device)

    if mask is None:
        mask = ~torch.isnan(Y)
    mask = mask.to(device)

    # Filtered targets for BCE
    y_obs = Y[mask].float()

    N, M = Y.shape
    model = LogisticMF(N, M, K).to(device)
    opt = LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        logits = model.forward()
        loss = F.binary_cross_entropy_with_logits(logits[mask], y_obs)
        loss.backward()
        return loss

    last_loss = None
    for t in range(steps):
        loss = opt.step(closure)
        last_loss = float(loss.detach())
        if verbose and (t % 25 == 0 or t == steps - 1):
            # print(f"step {t:3d} | nll = {last_loss:.6f}")
            pass

    return model


def compute_r(probs, Y, idtor):

    counts = idtor.sum(dim=1)
    keep = counts > 0
    
    P_sel = (probs * idtor)[keep]
    Y_sel = (Y * idtor)[keep]
    counts = counts[keep]
    mean_score_pred = P_sel.sum(dim=1) / counts
    mean_score_true = Y_sel.sum(dim=1) / counts

    x = mean_score_true.cpu().numpy()
    y = mean_score_pred.cpu().numpy()

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 2 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
        r = np.corrcoef(x[valid], y[valid])[0, 1]
    else:
        r = np.nan
    return r




    
def simple_model_job(dataset, masking_method, factor, trial_id):

    K_fit = factor
    i = trial_id
    masking_method = masking_method
    dataset = dataset
    
    os.makedirs("results/auc", exist_ok=True)
    os.makedirs("results/corr", exist_ok=True)
    config_name = f"{dataset}_{masking_method}_k{K_fit}_i{i}"

    run_name = f"wandb5_{config_name}"
    # wandb.login(key="575119bcea40be5839a138fbe59d95326bbeb2db")
    # wandb.init(
    #     project="info-ga-2",
    #     name=run_name,
    #     settings=wandb.Settings(
    #         save_code=False,     # already minimal
    #         init_timeout=300     # avoid 90s handshake timeout
    #     )
    #     # mode="offline",       # uncomment if your cluster egress is flaky; later `wandb sync`
    # )   
    train_corr_path = f"results/corr/train_corr_{config_name}.pt"
    test_corr_path = f"results/corr/test_corr_{config_name}.pt"

    # ---- check if both files exist ----
    if os.path.exists(train_corr_path) and os.path.exists(test_corr_path):      
        if torch.load(train_corr_path) is not None and torch.load(test_corr_path) is not None:
            print(f"[Skip] Both {train_corr_path} and {test_corr_path} already exist. Exiting.")
        
            return
        
    print(f"running rank:{K_fit} at trial: {i} ")
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)
    
    # data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_everything_benchmark(i, filter_method= "random_mask")
    if dataset == "HELM":
        if masking_method in ['date','size']:
            return
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_helm_benchmark(i, masking_method)
    elif dataset == "everything":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_everything_benchmark(i, masking_method)
    elif dataset == "official_provider":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_official_provider_benchmark(i, masking_method)
    else:
        assert False
    
    Y = data_with0.clone()
    N, M = Y.shape[0], Y.shape[1]
    
    Y_missing = Y.clone().float()
    
    Y_missing[~train_idtor.bool()] = float("nan")

    model = fit_logistic_mf(Y_missing, K=K_fit, mask=train_idtor, steps=50, lr=5e-3, device="cuda:0")
    with torch.no_grad():
        auroc = AUROC(task="binary")
        P_hat = torch.sigmoid(model.forward())
        P_hat = P_hat.cpu()
        train_auc = auroc(P_hat[train_idtor].cpu(), Y[train_idtor].cpu())
        print(f"factor {K_fit} train auc: {train_auc}")

        test_auc = auroc(P_hat[test_idtor].cpu(), Y[test_idtor].cpu())
        print(f"factor {K_fit} test auc: {test_auc}")


        torch.save(train_auc, f"results/auc/train_auc_{config_name}.pt")
        torch.save(test_auc, f"results/auc/test_auc_{config_name}.pt")

        r_train = compute_r(P_hat.cpu(), Y.cpu(), train_idtor.cpu())
        r_test = compute_r(P_hat.cpu(), Y.cpu(), test_idtor.cpu())
        
        print(f"{dataset} {masking_method} factor {K_fit} train corr: {r_train}")
        print(f"{dataset} {masking_method} factor {K_fit} test corr: {r_test}")
        print("*"*30)
        
        torch.save(r_train, f"results/corr/train_corr_{config_name}.pt")
        torch.save(r_test, f"results/corr/test_corr_{config_name}.pt")
        
        # run_results = {
        #     "train_auc": float(train_auc),
        #     "test_auc": float(test_auc),
        #     "train_corr": float(r_train) if r_train == r_train else None,  # None if NaN
        #     "test_corr": float(r_test) if r_test == r_test else None,
        # }   
        
        # wandb.log(run_results)
        # wandb.finish()


def sequential_job(factor, trial_id, dataset_list, masking_method_list):
    for masking_method in masking_method_list:
        for dataset in dataset_list:
            try:
                simple_model_job(dataset, masking_method, factor, trial_id)
            except Exception as e:
                print(f"err: dataset={dataset}, mask={masking_method}, factor={factor}, trial={trial_id}")
                print(f" -> {type(e).__name__}: {e}")        # just type and message
                traceback.print_exc()                        # full stack trace (optional)
        
def run_single_factor(factor, trial_id, dataset_list, masking_method_list):
    """Wrapper function for a single factor - this will be parallelized"""
    print(f"Starting factor {factor} on process {os.getpid()}")
    sequential_job(factor, trial_id, dataset_list, masking_method_list)
    print(f"Completed factor {factor}")
    return factor


# Approach 1: Using multiprocessing.Pool
def run_parallel_pool(trial_id, dataset_list, masking_method_list, factor_list):
    """Parallelize using multiprocessing.Pool"""
    
    # Create partial function with fixed arguments
    run_factor_partial = functools.partial(
        run_single_factor, 
        trial_id=trial_id,
        dataset_list=dataset_list,
        masking_method_list=masking_method_list
    )
    
    with Pool(processes=15) as pool:
        results = pool.map(run_factor_partial, factor_list)
    
    print(f"Completed all factors: {results}")

    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="HELM")
    # parser.add_argument("--masking_method", type=str, default='random_mask')
    # parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--trial_id", type=int, default=0)
    args = parser.parse_args()
    # parallelize this part
    mask_list = ["date","size","random_mask","random_row"] #"random_mask","random_row",
    dataset_list = ["official_provider"] #,"HELM"
    factor_list = [1, 2, 3, 8, 15, 30, 50]
    # simple_model_job("official_provider", "date", 2, 1)
    # parallelize the part below
    # sequential_job(0, 1, dataset_list, mask_list)
    # sequential_job(1, args.trial_id,dataset_list,mask_list)
    run_parallel_pool(args.trial_id, dataset_list, mask_list, factor_list)
    
    # for factor in factor_list:
    #     sequential_job(factor, args.trial_id)