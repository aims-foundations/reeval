import pandas as pd
import matplotlib.pyplot as plt
import torch
from goodness_of_fit import goodness_of_fit_1PL

if __name__ == "__main__":
    file_path = '../data/real/irt_result/normal_mmlu/theta/pyMLE_mask_1PL_theta_manual.csv'
    df = pd.read_csv(file_path)
    x = df.iloc[:, 1]
    y = df.iloc[:, 2]
    corr = x.corr(y)
    
    plt.scatter(x, y)
    plt.xlabel('theta from python MLE')
    plt.ylabel('mmlu score')
    plt.title(f'Correlation: {corr:.2f}')
    plt.savefig('../plot/real/theta_corr_pyMLE.png', dpi=300, bbox_inches='tight')

    # Goodness of fit
    Z_df = pd.read_csv('../data/real/irt_result/normal_mmlu/Z/pyMLE_mask_1PL_Z.csv')
    Z = Z_df.loc[:, "z3"].values

    theta_df = pd.read_csv('../data/real/irt_result/normal_mmlu/theta/pyMLE_mask_1PL_theta.csv')
    theta = theta_df.loc[:, "theta"].values

    y_df = pd.read_csv('../data/real/response_matrix/normal_mmlu/non_mask_matrix.csv', index_col=0)

    goodness_of_fit_1PL(
        Z=Z,
        theta=theta,
        y_df=y_df,
        plot_path='../plot/real/pyMLE_goodness_of_fit.png',
        bin_size=7,
    )
    