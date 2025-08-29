import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torchmetrics import AUROC
from grab_data import load_old_benchmark, get_new_benchmark
import numpy as np
import os
import matplotlib.pyplot as plt

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
    
    print("running experiment")
    factors = [i for i in range(1,16)]
    num_trials = 100
    train_auc_table = np.zeros((len(factors), num_trials), dtype=np.float64)
    test_auc_table  = np.zeros((len(factors), num_trials), dtype=np.float64)
    print("running experiment")
    os.makedirs("results", exist_ok=True)
    for K_fit in factors:
        for i in range(num_trials):
            try:
                print(f"running rank:{K_fit} at trial: {i} ")
                torch.manual_seed(i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(i)
                # data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_new_benchmark(i)
                data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = load_old_benchmark(i)
                Y = data_with0
                N, M = Y.shape[0], Y.shape[1]
                
                Y_missing = Y.clone().float()
                
                Y_missing[~train_idtor.bool()] = float("nan")

                model = fit_logistic_mf(Y_missing, K=K_fit, mask=train_idtor, steps=50, lr=5e-3, device="cuda:1")
                with torch.no_grad():
                    auroc = AUROC(task="binary")
                    P_hat = torch.sigmoid(model.forward())

                    train_auc = auroc(P_hat[train_idtor].cpu(), Y[train_idtor].cpu())
                    print(f"factor {K_fit} train auc: {train_auc}")

                    test_auc = auroc(P_hat[test_idtor].cpu(), Y[test_idtor].cpu())
                    print(f"factor {K_fit} test auc: {test_auc}")
                    train_auc_table[K_fit-1][i] = train_auc.item()
                    test_auc_table[K_fit-1][i] = test_auc.item()
                    # mae = torch.mean(abs(P_hat.cpu() - P))
                    # print(f"MSE on P: {mae}")
            except:
                print(f"crahsed at rank:{K_fit} trial {i}")
                continue
                    
                
            print("*"*20)
            
            
            # np.savetxt("results/train_auc_table.csv", train_auc_table, delimiter=",", fmt="%.6f")
            # np.savetxt("results/test_auc_table.csv",  test_auc_table,  delimiter=",", fmt="%.6f")