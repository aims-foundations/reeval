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
    
    # --- setup 4x4 grid ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    i=0
    for idx, K_fit in enumerate(range(1, 17)):
        ax = axes[idx]

        print(f"running rank:{K_fit} at trial: {i} ")
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(i)

        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, cat = load_old_benchmark(i)
        Y = data_with0
        N, M = Y.shape[0], Y.shape[1]
        
        Y_missing = Y.clone().float()
        Y_missing[~train_idtor.bool()] = float("nan")

        model = fit_logistic_mf(Y_missing, K=K_fit, mask=train_idtor, steps=50, lr=5e-3, device="cuda:3")
        with torch.no_grad():
            auroc = AUROC(task="binary")
            P_hat = torch.sigmoid(model.forward())

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
        title_r = f"{r:.3f}" if np.isfinite(r) else "NA"

        # scatter plot
        ax.scatter(x, y, alpha=0.6, s=10)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"Rank={K_fit}, r={title_r}")
        ax.tick_params(labelsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.savefig("plot/test_set_mean/all_ranks_1to16.png", dpi=300, bbox_inches="tight")
    plt.show()