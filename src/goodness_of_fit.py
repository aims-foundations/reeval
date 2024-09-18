import warnings
import pandas as pd
import torch
import numpy as np
from utils import item_response_fn_1PL
import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')

def goodness_of_fit_1PL(
    Z,
    theta,
    y_df,
    plot_path,
    bin_size=7,
):
    assert y_df.shape[1] == len(Z), f"Number of columns in y_df ({y_df.shape[1]}) does not match the length of Z ({len(Z)})"
    assert y_df.shape[0] == len(theta), f"Number of rows in y_df ({y_df.shape[0]}) does not match the length of theta ({len(theta)})"

    theta = torch.tensor(theta, dtype=torch.float32)
    if torch.isnan(theta).any():
        warnings.warn("Warning: 'theta' contains NaN values.")
    theta_no_nan = theta[~torch.isnan(theta)]
    bin_start = torch.min(theta_no_nan)
    bin_end = torch.max(theta_no_nan)
    bins = np.linspace(bin_start, bin_end, bin_size)
    print(bins)
    # [-3. -2. -1.  0.  1.  2.  3.]

    diff_list = []
    for i in range(len(Z)):
        single_z3 = torch.tensor(Z[i], dtype=torch.float32)

        y_col = y_df.iloc[:, i].values

        for j in range(len(bins) - 1):
            bin_mask = (theta >= bins[j]) & (theta < bins[j + 1])
            if bin_mask.sum() > 0: # bin not empty
                y_empirical = y_col[(bin_mask) & (y_col != -1)].mean()

                theta_mid = (bins[j] + bins[j + 1]) / 2
                theta_mid_tensor = torch.tensor([theta_mid], dtype=torch.float32)
                y_theoretical = item_response_fn_1PL(theta_mid_tensor, single_z3).item()

                diff = abs(y_empirical - y_theoretical)
                diff_list.append(diff)

    diff_array = np.array(diff_list)
    mean_diff = diff_array.mean()
    std_diff = diff_array.std()

    print(f'Mean of differences: {mean_diff}')
    print(f'Standard deviation of differences: {std_diff}')

    plt.figure(figsize=(10, 6))
    plt.hist(diff_list, bins=40, density=True, alpha=0.4)
    plt.xlabel(r'Difference between empirical and theoretical $P(y=1)$', fontsize=30)
    plt.tick_params(axis='both', labelsize=25)
    plt.xlim(0, 1)
    plt.axvline(mean_diff, linestyle='--')
    plt.text(mean_diff, plt.gca().get_ylim()[1], f'{mean_diff:.2f}', 
            ha='center', va='bottom', fontsize=25)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    Z_df = pd.read_csv('../data/real/irt_result/normal/Z/all_1PL_Z_clean.csv')
    Z = Z_df.loc[:, "z3"].values

    theta_df = pd.read_csv('../data/real/irt_result/normal/theta/all_1PL_theta.csv')
    theta = theta_df.loc[:, "F1"].values

    y_df = pd.read_csv('../data/real/response_matrix/normal/all_matrix.csv', index_col=0)

    goodness_of_fit_1PL(Z, theta, y_df, plot_path='../plot/real/goodness_of_fit.png')