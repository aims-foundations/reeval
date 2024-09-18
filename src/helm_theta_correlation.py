import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')
from goodness_of_fit import goodness_of_fit_1PL
from utils import set_seed

def theta_corr_plot(
    x,
    y,
    plot_path,
):
    corr = np.corrcoef(x, y)[0, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.xlabel(r'$\theta$ from IRT calibration', fontsize=45)
    plt.ylabel(r'CTT score from leaderboard', fontsize=45)
    plt.title(f'Correlation: {corr:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    
    set_seed(42)
    if args.dataset == 'mmlu':
        theta_corr_path = '../data/real/irt_result/normal_mmlu/theta/pyMLE_mask_1PL_theta_manual.csv'
        Z_path = '../data/real/irt_result/normal_mmlu/Z/pyMLE_mask_1PL_Z.csv'
        theta_path = '../data/real/irt_result/normal_mmlu/theta/pyMLE_mask_1PL_theta.csv'
        y_path = '../data/real/response_matrix/normal_mmlu/non_mask_matrix.csv'
        theta_col_name = "theta"
    elif args.dataset == 'airbench':
        theta_corr_path = '../data/real/irt_result/normal/theta/all_1PL_theta_manual.csv'
        Z_path = '../data/real/irt_result/normal/Z/all_1PL_Z_clean.csv'
        theta_path = '../data/real/irt_result/normal/theta/all_1PL_theta.csv'
        y_path = '../data/real/response_matrix/normal/all_matrix.csv'
        theta_col_name = "F1"
    elif args.dataset == 'syn_rea':
        theta_corr_path = '../data/real/irt_result/pyMLE_normal_syn_reason/theta/mask_1PL_theta_manual.csv'
        Z_path = '../data/real/irt_result/pyMLE_normal_syn_reason/Z/mask_1PL_Z.csv'
        theta_path = '../data/real/irt_result/pyMLE_normal_syn_reason/theta/mask_1PL_theta.csv'
        y_path = '../data/real/response_matrix/normal_syn_reason/mask_matrix.csv'
        theta_col_name = "theta"
    
    df = pd.read_csv(theta_corr_path)
    x = df.loc[:, theta_col_name].values
    y = df.loc[:, "score"].values
    x = np.nan_to_num(x, nan=0)
    
    theta_corr_plot(
        x=x,
        y=y,
        plot_path=f'../plot/real/{args.dataset}_theta_corr.png',
    )

    # Goodness of fit
    Z_df = pd.read_csv(Z_path)
    Z = Z_df.loc[:, "z3"].values

    theta_df = pd.read_csv(theta_path)
    theta = theta_df.loc[:, theta_col_name].values

    y_df = pd.read_csv(y_path, index_col=0)

    goodness_of_fit_1PL(
        Z=Z,
        theta=theta,
        y_df=y_df,
        plot_path=f'../plot/real/{args.dataset}_pyMLE_goodness_of_fit.png',
        bin_size=7,
    )
    