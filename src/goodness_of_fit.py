import pandas as pd
import torch
import numpy as np
from utils import item_response_fn_1PL
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_coef_1PL = pd.read_csv('../data/real/irt_result/Z/all_1PL_Z_clean.csv')
    all_value_1PL = all_coef_1PL.iloc[:, 2].values

    theta_df = pd.read_csv('../data/real/irt_result/theta/all_1PL_theta.csv')
    theta = torch.tensor(theta_df.iloc[:, 1].values, dtype=torch.float32)

    y_df = pd.read_csv('../data/real/response_matrix/all_matrix.csv')

    assert y_df.shape[1] == len(all_value_1PL), f"Number of columns in y_df ({y_df.shape[1]}) does not match the length of all_value_1PL ({len(all_value_1PL)})"
    assert y_df.shape[0] == len(theta_df), f"Number of rows in y_df ({y_df.shape[0]}) does not match the length of theta_df ({len(theta_df)})"

    bins = np.linspace(-3, 3, 7)
    print(bins)
    # [-3. -2. -1.  0.  1.  2.  3.]

    diff_list = []
    for i in range(len(all_value_1PL)):
        single_z3 = torch.tensor(all_value_1PL[i], dtype=torch.float32)

        y_col = y_df.iloc[:, i].values

        for j in range(len(bins) - 1):
            bin_mask = (theta >= bins[j]) & (theta < bins[j + 1])
            if bin_mask.sum() > 0: # bin not empty
                y_empirical = y_col[bin_mask].mean()

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
    plt.hist(diff_list, bins=40, density=True, alpha=0.7, color='blue')
    plt.xlabel('Difference')
    plt.ylabel('Density')
    plt.title('Histogram of Differences (Empirical vs Theoretical)')
    plt.grid(True)
    plt.savefig('../plot/real/goodness_of_fit.png')