from matplotlib import gridspec
import torch
import numpy as np
import random
import jax.numpy as jnp
import warnings
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')

def item_response_fn_1PL(z3, theta):
    return 1 / (1 + torch.exp(-(theta + z3)))

def item_response_fn_1PL_jnp(z3, theta):
    return 1 / (1 + jnp.exp(-(theta + z3)))
    
def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def goodness_of_fit_1PL(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bin_size: int=7,
):
    assert y.shape[1] == z.shape[0], f'{y.shape[1]} != {z.shape[0]}'
    assert y.shape[0] == theta.shape[0], f'{y.shape[0]} != {theta.shape[0]}'

    theta = theta.detach().cpu()
    bin_start = torch.min(theta)
    bin_end = torch.max(theta)
    bins = torch.linspace(bin_start, bin_end, bin_size)
    print(bins) # [-3. -2. -1.  0.  1.  2.  3.]

    diff_list = []
    for i in range(z.shape[0]):
        single_z = z[i]
        y_col = y[:, i]

        for j in range(bins.shape[0] - 1):
            bin_mask = (theta >= bins[j]) & (theta < bins[j + 1])
            if bin_mask.sum() > 0: # bin not empty
                y_empirical = y_col[(bin_mask) & (y_col != -1)].mean()

                theta_mid = (bins[j] + bins[j + 1]) / 2
                y_theoretical = item_response_fn_1PL(theta_mid, single_z).item()

                diff = abs(y_empirical - y_theoretical)
                diff_list.append(diff)

    diff_array = np.array(diff_list)
    mean_diff = np.mean(diff_array)
    std_diff = np.std(diff_array)
    print(f'Mean of differences: {mean_diff}')
    print(f'Standard deviation of differences: {std_diff}')

    plt.figure(figsize=(10, 6))
    plt.hist(diff_list, bins=40, density=True, alpha=0.4)
    plt.xlabel(r'Difference between empirical and theoretical $P(y=1)$', fontsize=30)
    plt.tick_params(axis='both', labelsize=25)
    plt.xlim(0, 1)
    plt.axvline(mean_diff, linestyle='--')
    plt.text(mean_diff, plt.gca().get_ylim()[1], f'{mean_diff:.2f}', ha='center', va='bottom', fontsize=25)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    return mean_diff, std_diff

def bootstrap_mean_std(data, ratio = 0.9, num_samples=1000):
    data = np.array(data)
    mean = np.mean(data)
    
    bootstrap_means = []
    for _ in range(num_samples):
        bootstrap_sample = np.random.choice(data, size=int(ratio * data.shape[0]), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
        
    std_bootstrap = np.std(bootstrap_means)
    return mean, std_bootstrap
    
def perform_t_test(sample_1, sample_2, label=""):
    print(f"{label} T-test:")
    print(f"Null Hypothesis (H0): The means of the two samples are equal.")
    print(f"Alternative Hypothesis (H1): The means of the two samples are not equal.")
    t_stat, p_value = ttest_ind(sample_1, sample_2)
    print(f"t_stat = {t_stat}, p_value = {p_value}")
    if p_value < 0.05:
        print(f"Reject the null hypothesis for {label}.")
    else:
        print(f"Fail to reject the null hypothesis for {label}.")

def theta_corr_ctt(
    theta: np.array,
    y: np.array,
    plot_path: str,
):
    assert y.shape[1] == theta.shape[0], f'{y.shape[1]} != {theta.shape[0]}'
    
    ctt_scores = []
    for row in y:
        valid_values = row[row != -1]
        if len(valid_values) > 0:
            ctt_scores.append(np.mean(valid_values))
        else:
            ctt_scores.append(np.nan)
    ctt_scores = np.array(ctt_scores)
    assert ctt_scores.shape[0] == theta.shape[0]
    
    if np.isnan(ctt_scores).any():
        warnings.warn("ctt_scores contains NaN values.", UserWarning)
    mask = ~np.isnan(ctt_scores)
    ctt_scores_masked = ctt_scores[mask]
    theta_masked = theta[mask]
    corr = np.corrcoef(ctt_scores_masked, theta_masked)[0, 1]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(theta_masked, ctt_scores_masked)
    plt.xlabel(r'$\theta$ from calibration', fontsize=45)
    plt.ylabel(r'CTT score', fontsize=45)
    plt.title(f'Correlation: {corr:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    return corr
    
    
    

def z_corr_plot(
    x,
    y,
    plot_path,
):
    corr = np.corrcoef(x, y)[0, 1]
    mse = np.mean((x - y) ** 2)
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.xlabel(r'$z$ from amortized IRT calibration', fontsize=45)
    plt.ylabel(r'$z$ from non-amortized IRT calibration', fontsize=45)
    plt.title(f'Correlation: {corr:.2f}, MSE: {mse:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
def theta_corr_plot(
    x,
    y,
    plot_path,
):
    corr = np.corrcoef(x, y)[0, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.xlabel(r'$\theta$ from calibration', fontsize=45)
    plt.ylabel(r'CTT score', fontsize=45)
    plt.title(f'Correlation: {corr:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
def plot_scatter_with_histograms(z3_py, z3_r, save_path, x_label=r'Our $z_3$', y_label=r'mirt $z_3$'):
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

    # Scatter plot between z3_py and z3_r
    ax_main = plt.subplot(gs[1, 0])
    ax_main.scatter(z3_py, z3_r)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    # Calculate correlation and add title at the bottom
    corr_np = np.corrcoef(z3_py, z3_r)[0, 1]
    plt.figtext(0.5, 0.02, f'Correlation: {corr_np:.2f}', ha='center')

    # Histogram for z3_py (top)
    ax_xhist = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_xhist.hist(z3_py, bins=30, color='gray', alpha=0.7)
    ax_xhist.axis('off')

    # Histogram for z3_r (right)
    ax_yhist = plt.subplot(gs[1, 1], sharey=ax_main)
    ax_yhist.hist(z3_r, bins=30, color='gray', alpha=0.7, orientation='horizontal')
    ax_yhist.axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()