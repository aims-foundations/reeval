import os
import numpy as np
import matplotlib.pyplot as plt

def load_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.loadtxt(path, delimiter=",")
    # Ensure 2D even if saved as single row
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def summarize_rows(table):
    """
    table: shape [num_factors, num_trials]
    returns: x, mean, p5, p95 (each shape [num_factors])
    """
    mean = np.nanmean(table, axis=1)
    p5   = np.nanpercentile(table, 2.5,  axis=1)
    p95  = np.nanpercentile(table, 97.5, axis=1)
    x = np.arange(1, table.shape[0] + 1)  # factors 1..num_factors
    return x, mean, p5, p95

def plot_with_ci(x, mean, p5, p95, title, outfile,y_lim_l = 0):
    plt.figure(figsize=(8, 5))
    # Shaded CI band
    plt.fill_between(x, p5, p95, alpha=0.2, label="5–95% range")
    # Mean with asymmetric error bars
    yerr = np.vstack([mean - p5, p95 - mean])
    plt.errorbar(x, mean, yerr=yerr, fmt="o-", capsize=3, linewidth=1.5, label="Mean AUROC")
    plt.xlabel("Factor rank (K)")
    plt.ylabel("AUROC")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xticks(x)  # show every factor on x-axis
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.ylim(y_lim_l, 1.0)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    train_path = "results/train_auc_table.csv"
    test_path  = "results/test_auc_table.csv"

    train_tbl = load_table(train_path)
    train_tbl = train_tbl[:10,:]
    test_tbl  = load_table(test_path)
    test_tbl = test_tbl[:10,:]

    # Summaries
    x_tr, m_tr, p5_tr, p95_tr = summarize_rows(train_tbl)
    x_te, m_te, p5_te, p95_te = summarize_rows(test_tbl)

    # Plots
    plot_with_ci(x_tr, m_tr, p5_tr, p95_tr, "Train AUROC vs Factor Rank (mean ± 2.5–97.5%)",
                 "plot/train_auc_summary_1.png")
    plot_with_ci(x_te, m_te, p5_te, p95_te, "Test AUROC vs Factor Rank (mean ± 2.5–97.5%)",
                 "plot/test_auc_summary_1.png")
