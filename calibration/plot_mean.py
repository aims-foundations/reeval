import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torchmetrics import AUROC
from util import get_helm_benchmark, get_official_provider_benchmark
import numpy as np
import os
import matplotlib.pyplot as plt
from tueplots import bundles
import argparse

bundles.iclr2024()
print("running experiment")
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

if __name__ == "__main__":
    # factor
    # dateset
    # masking
    # trial
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HELM")
    parser.add_argument("--masking_method", type=str, default='random_mask')
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--trial_id", type=int, default=0)
    args = parser.parse_args()
    K_fit = args.factor
    i = args.trial_id
    masking_method = args.masking_method
    dataset = args.dataset
    
    os.makedirs("results/auc", exist_ok=True)
    


    print(f"running rank:{K_fit} at trial: {i} ")
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)

    # data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, cat = load_old_benchmark(i)
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, cat = get_official_provider_benchmark(i)
    Y = data_with0
    N, M = Y.shape[0], Y.shape[1]
    model_names = cat[2]
    Y_missing = Y.clone().float()
    Y_missing[~train_idtor.bool()] = float("nan")

    model = fit_logistic_mf(Y_missing, K=K_fit, mask=train_idtor, steps=50, lr=5e-3, device="cuda:3")
    with torch.no_grad():
        auroc = AUROC(task="binary")
        P_hat = torch.sigmoid(model.forward())
        P_hat = P_hat.cpu()
        test_auc = auroc(P_hat[test_idtor].cpu(), Y[test_idtor].cpu())
        
        mask_test = test_idtor

        # Predicted and true values
        P_sel = P_hat * mask_test
        Y_sel = Y * mask_test
        counts = mask_test.sum(dim=1)
        testtaker_mean_score_pred = P_sel.sum(dim=1) / counts
        testtaker_mean_score_true = Y_sel.sum(dim=1) / counts

    x = testtaker_mean_score_true.cpu().numpy()
    y = testtaker_mean_score_pred.cpu().numpy()

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 2 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
        r = np.corrcoef(x[valid], y[valid])[0, 1]
    else:
        r = np.nan

    torch.save(train_auc, f"results/auc/train_auc_{dataset}_{masking_method}_k{K_fit}_i{i}.pt")
    torch.save(test_auc, f"results/auc/test_auc_{dataset}_{masking_method}_k{K_fit}_i{i}.pt")
