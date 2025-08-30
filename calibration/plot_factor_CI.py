import os
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.iclr2024()
print("running experiment")

def load_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.loadtxt(path, delimiter=",")
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

def plot_two_datasets_with_ci(
    x1, mtr1, p5tr1, p95tr1, mte1, p5te1, p95te1, label1,
    x2, mtr2, p5tr2, p95tr2, mte2, p5te2, p95te2, label2,
    outfile, title="AUROC vs Factor Rank", y_lim_l=0.0
):
    
    # plt.rcParams.update({
    #     "font.size": 60,       # default font size
    #     # "axes.titlesize": 16,  # title
    #     # "axes.labelsize": 14,  # x and y labels
    #     "xtick.labelsize": 30,
    #     "ytick.labelsize": 60,
    #     "legend.fontsize": 12
    # })    
    with plt.rc_context(bundles.iclr2024(usetex=True, family="serif")):
        plt.figure(figsize=(5, 4))

        plt.rcParams.update({
            "font.size": 60,       # default font size
            # "axes.titlesize": 16,  # title
            # "axes.labelsize": 14,  # x and y labels
            # "xtick.labelsize": 30,
            # "ytick.labelsize": 60,
            # "legend.fontsize": 12
        })    

        # Two distinct colors from the default cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c1 = colors[0] if len(colors) > 0 else "C0"
        c2 = colors[1] if len(colors) > 1 else "C1"

        # --- Dataset 1 (color c1): dashed=train, solid=test ---
        plt.fill_between(x1, p5tr1, p95tr1, alpha=0.10, color=c1)
        plt.plot(x1, mtr1, linestyle="--", marker="o", linewidth=1.6, color=c1,
                label=f"{label1} – Train")

        plt.fill_between(x1, p5te1, p95te1, alpha=0.15, color=c1)
        plt.plot(x1, mte1, linestyle="-", marker="o", linewidth=1.8, color=c1,
                label=f"{label1} – Test")

        # --- Dataset 2 (color c2): dashed=train, solid=test ---
        plt.fill_between(x2, p5tr2, p95tr2, alpha=0.10, color=c2)
        plt.plot(x2, mtr2, linestyle="--", marker="s", linewidth=1.6, color=c2,
                label=f"{label2} – Train")

        plt.fill_between(x2, p5te2, p95te2, alpha=0.15, color=c2)
        plt.plot(x2, mte2, linestyle="-", marker="s", linewidth=1.8, color=c2,
                label=f"{label2} – Test")

        # Axes, legend, etc.
        xticks = np.unique(np.concatenate([x1, x2]))
        plt.xticks(xticks)
        plt.xlabel("Factor rank (K)")
        plt.ylabel("AUROC")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.75, 1.0)
        plt.legend(ncol=2, frameon=True)
        plt.tight_layout()

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=600,  bbox_inches="tight")
        plt.close()
        print(f"Saved: {outfile}")

if __name__ == "__main__":
    # --- Dataset A (original) ---
    train_path_A = "results/train_auc_table.csv"
    test_path_A  = "results/test_auc_table.csv"

    train_tbl_A = load_table(train_path_A)[:10, :]
    test_tbl_A  = load_table(test_path_A)[:10, :]

    x_tr_A, m_tr_A, p5_tr_A, p95_tr_A = summarize_rows(train_tbl_A)
    x_te_A, m_te_A, p5_te_A, p95_te_A = summarize_rows(test_tbl_A)

    # If train/test have different K, align to common K for plotting
    K_A = min(len(x_tr_A), len(x_te_A))
    xA  = np.arange(1, K_A + 1)
    m_tr_A, p5_tr_A, p95_tr_A = m_tr_A[:K_A], p5_tr_A[:K_A], p95_tr_A[:K_A]
    m_te_A, p5_te_A, p95_te_A = m_te_A[:K_A], p5_te_A[:K_A], p95_te_A[:K_A]

    # --- Dataset B (new) ---
    train_path_B = "results/train_auc_table_new.csv"
    test_path_B  = "results/test_auc_table_new.csv"

    train_tbl_B = load_table(train_path_B)[:10, :]
    test_tbl_B  = load_table(test_path_B)[:10, :]

    x_tr_B, m_tr_B, p5_tr_B, p95_tr_B = summarize_rows(train_tbl_B)
    x_te_B, m_te_B, p5_te_B, p95_te_B = summarize_rows(test_tbl_B)

    K_B = min(len(x_tr_B), len(x_te_B))
    xB  = np.arange(1, K_B + 1)
    m_tr_B, p5_tr_B, p95_tr_B = m_tr_B[:K_B], p5_tr_B[:K_B], p95_tr_B[:K_B]
    m_te_B, p5_te_B, p95_te_B = m_te_B[:K_B], p5_te_B[:K_B], p95_te_B[:K_B]

    # --- Single combined figure ---
    plot_two_datasets_with_ci(
        xA, m_tr_A, p5_tr_A, p95_tr_A, m_te_A, p5_te_A, p95_te_A, "Dataset A",
        xB, m_tr_B, p5_tr_B, p95_tr_B, m_te_B, p5_te_B, p95_te_B, "Dataset B",
        outfile="plot/auc_summary_both.png",
        title="Train/Test AUROC vs Factor Rank (two datasets)",
        y_lim_l=0.0,
    )