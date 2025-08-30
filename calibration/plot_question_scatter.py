import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torchmetrics import AUROC
from grab_data import load_old_benchmark, get_new_benchmark
import numpy as np
import os
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.iclr2024()

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
    
    factors = [i for i in range(1,16)]
    num_trials = 100
    train_auc_table = np.zeros((len(factors), num_trials), dtype=np.float64)
    test_auc_table  = np.zeros((len(factors), num_trials), dtype=np.float64)

    os.makedirs("results", exist_ok=True)
    K_fit = 2
    i  = 0
    cat_level = 1
    
    print(f"running rank:{K_fit} at trial: {i} ")
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)
    # data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_new_benchmark(i)
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor,cat = load_old_benchmark(i)
    Y = data_with0
    N, M = Y.shape[0], Y.shape[1]
    
    Y_missing = Y.clone().float()
    
    Y_missing[~train_idtor.bool()] = float("nan")

    model = fit_logistic_mf(Y_missing, K=K_fit, mask=train_idtor, steps=50, lr=5e-3, device="cuda:4")

    with torch.no_grad():
        auroc = AUROC(task="binary")
        P_hat = torch.sigmoid(model.forward())

        train_auc = auroc(P_hat[train_idtor].cpu(), Y[train_idtor].cpu())
        print(f"factor {K_fit} train auc: {train_auc}")

        test_auc = auroc(P_hat[test_idtor].cpu(), Y[test_idtor].cpu())
        
        print(f"factor {K_fit} test auc: {test_auc}")

        mask_test = test_idtor
        # Predicted and true values
        P_sel = P_hat * mask_test       # zero out masked entries
        Y_sel = Y * mask_test
        
        counts = mask_test.sum(dim=1)
        
        

        # Mean across columns, ignoring masked-out entries
        testtaker_mean_score_pred = P_sel.sum(dim=1) / counts
        testtaker_mean_score_true = Y_sel.sum(dim=1) / counts
    
    x = testtaker_mean_score_true.cpu().numpy()
    y = testtaker_mean_score_pred.cpu().numpy()
    
    
    
    V = model.V.detach().cpu().numpy()   # shape (78712, 2)
    labels = np.array(cat[cat_level])            # convert list to numpy array

    # Get unique categories and map them to colors
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))  # up to 10 distinct, change if more

    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    color_indices = [label_to_idx[l] for l in labels]
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        # Plot scatter
        
        plt.rcParams.update({
            "font.size": 12,       # default font size
            "axes.titlesize": 16,  # title
            "axes.labelsize": 14,  # x and y labels
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12
        })    
        
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(V[:, 0], V[:, 1], c=color_indices, cmap=colors, s=2, alpha=0.7)

        # Add legend
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors(i)) 
                for i in range(len(unique_labels))]
        plt.legend(handles, unique_labels, title="Category", bbox_to_anchor=(1.05, 1), frameon=True)

        plt.xlabel("V[:, 0]")
        plt.ylabel("V[:, 1]")
        plt.title(f"Scatter of model.V colored by category seed={i} level = {cat_level} Old Dataset")
        plt.tight_layout()
        plt.show()
        save_path = f"plot/visualize_factor_matrix_seed{i}_lvl{cat_level}.png"
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()  
        print(f"save to {save_path}")