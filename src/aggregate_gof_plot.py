import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')
from utils import PLOT_NAME_MAP, DATASETS

if __name__ == "__main__":
    plot_dir = '../plot/aggregate_gof_plot'
    os.makedirs(plot_dir, exist_ok=True)
    
    trad_gof = pd.read_csv(f'../plot/nonamor_calibration/nonamor_calibration_gof.csv')['gof_means'].values
    plugin_gof_train = pd.read_csv(f'../plot/plugin_regression/plugin_regression_gof_train.csv')['gof_means'].values
    plugin_gof_test = pd.read_csv(f'../plot/plugin_regression/plugin_regression_gof_test.csv')['gof_means'].values
    joint_gof_train = pd.read_csv(f'../plot/amor_calibration/amor_calibration_gof_train.csv')['gof_means'].values
    joint_gof_test = pd.read_csv(f'../plot/amor_calibration/amor_calibration_gof_test.csv')['gof_means'].values
    dim2_1pl_trad_gof = pd.read_csv(f'../plot/mle_multi_dim_calibration/dim2_1pl_gof_con_True.csv')['gof_means'].values
    dim2_1pl_amor_gof_train_dataset = pd.read_csv(f'../plot/mle_multi_dim_amor_theta/mle_multi_dim_amor_theta_gof_con_True_train.csv')['datasets'].values
    dim2_1pl_amor_gof_train = pd.read_csv(f'../plot/mle_multi_dim_amor_theta/mle_multi_dim_amor_theta_gof_con_True_train.csv')['gof_means'].values
    dim2_1pl_amor_gof_test_dataset = pd.read_csv(f'../plot/mle_multi_dim_amor_theta/mle_multi_dim_amor_theta_gof_con_True_test.csv')['datasets'].values
    dim2_1pl_amor_gof_test = pd.read_csv(f'../plot/mle_multi_dim_amor_theta/mle_multi_dim_amor_theta_gof_con_True_test.csv')['gof_means'].values
    
    dim2_1pl_amor_gof_train_aligned = [None] * len(DATASETS)
    for dataset, gof in zip(dim2_1pl_amor_gof_train_dataset, dim2_1pl_amor_gof_train):
        idx = DATASETS.index(dataset)
        dim2_1pl_amor_gof_train_aligned[idx] = gof
    
    dim2_1pl_amor_gof_test_aligned = [None] * len(DATASETS)
    for dataset, gof in zip(dim2_1pl_amor_gof_test_dataset, dim2_1pl_amor_gof_test):
        idx = DATASETS.index(dataset)
        dim2_1pl_amor_gof_test_aligned[idx] = gof
        
    datasets = [PLOT_NAME_MAP[dataset] for dataset in DATASETS]
    sorted_data = sorted(
        zip(datasets, trad_gof, plugin_gof_train, plugin_gof_test, joint_gof_train, joint_gof_test),
        key=lambda x: x[1]
    )
    datasets, trad_gof, plugin_gof_train, plugin_gof_test, joint_gof_train, joint_gof_test = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(datasets))

    plt.plot(x, trad_gof, 'k-', marker='o', label='Traditional')
    # plt.plot(x, plugin_gof_train, 'b-', marker='o', label='Plug-in (Train)')
    # plt.plot(x, plugin_gof_test, 'b--', marker='o', label='Plug-in (Test)')
    # plt.plot(x, joint_gof_train, 'r-', marker='o', label='Joint (Train)')
    # plt.plot(x, joint_gof_test, 'r--', marker='o', label='Joint (Test)')
    plt.plot(x, dim2_1pl_trad_gof, 'g-', marker='o', label='2D 1PL Traditional')
    plt.plot(x, dim2_1pl_amor_gof_train_aligned, 'purple', linestyle='-', marker='o', label='2D 1PL Amortized Train')
    plt.plot(x, dim2_1pl_amor_gof_test_aligned, 'purple', linestyle='--', marker='o', label='2D 1PL Amortized Test')
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2)
    
    plt.tick_params(axis='both', labelsize=25)
    plt.xticks(x, datasets, rotation=45, ha='right', fontsize=20)
    plt.xlabel('Datasets', fontsize=25)
    plt.ylabel('Goodness of Fit', fontsize=25)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=10)
    plt.savefig(f'{plot_dir}/agg_GOF.png', bbox_inches='tight', dpi=300)