import os
import torch
import pandas as pd
from utils import goodness_of_fit_1PL, theta_corr_ctt

if __name__ == "__main__":
    plot_dir = f'../plot/nonamor_calibration'
    os.makedirs(plot_dir, exist_ok=True)
    
    input_dir = '../data/nonamor_calibration/'
    datasets = [f for f in os.listdir(input_dir)]
    
    for dataset in datasets:
        y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0).values
        theta_hat = pd.read_csv(f'{input_dir}/{dataset}/nonamor_theta.csv')['theta'].values
        z_hat = pd.read_csv(f'{input_dir}/{dataset}/nonamor_z.csv')['z'].values
        
        mean_diff, std_diff = goodness_of_fit_1PL(
            z=torch.tensor(z_hat, dtype=torch.float32),
            theta=torch.tensor(theta_hat, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            plot_path=f"{plot_dir}/goodness_of_fit_{dataset}",
        )
        
        corr_ctt = theta_corr_ctt(
            theta=theta_hat,
            y=y,
            plot_path=f"{plot_dir}/theta_corr_ctt_{dataset}",
        )
        
        

        
        
        