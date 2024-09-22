import os
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset,  concatenate_datasets
from utils import goodness_of_fit_1PL, theta_corr_ctt, error_bar_plot

def main(
    hf_repo,
    y_path,
    df_z_train_path,
    df_z_test_path,
    df_theta_path,
):
    y = pd.read_csv(y_path, index_col=0).values
    y = torch.tensor(y, dtype=torch.float32)
    
    dataset_train = load_dataset(hf_repo, split="train")
    dataset_test = load_dataset(hf_repo, split="test")
    dataset = concatenate_datasets([dataset_train, dataset_test])
    emb = torch.tensor(dataset['embed'], dtype=torch.float32)
    
    assert y.shape[1] == emb.shape[0]
    train_indices, test_indices = split_indices(emb.shape[0])    
    emb_train = emb[train_indices]
    y_train = y[:, train_indices]
    emb_test = emb[test_indices]
    
    theta_train, z_train, W_train = amor_calibration(y_train, emb_train)
    z_test = torch.matmul(emb_test, W_train.cpu().detach())
    
    df_z_train = pd.DataFrame({
        'index': train_indices,
        'z': z_train.cpu().detach().numpy(),
    })
    df_z_train.to_csv(df_z_train_path, index=False)
    
    df_z_test = pd.DataFrame({
        'index': test_indices,
        'z': z_test.cpu().detach().numpy(),
    })
    df_z_test.to_csv(df_z_test_path, index=False)
    
    df_theta = pd.DataFrame({
        'theta': theta_train.cpu().detach().numpy(),
    })
    df_theta.to_csv(df_theta_path, index=False)

if __name__ == "__main__":
    input_dir = '../data/amor_calibration/'
    datasets = [f for f in os.listdir(input_dir)]
    
    plot_dir = f'../plot/amor_calibration'
    os.makedirs(plot_dir, exist_ok=True)
    
    gof_means, gof_stds, corr_ctt_means, corr_ctt_stds = [], [], [], []
    for dataset in tqdm(datasets):
        for i in range(10):
            y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0).values
            theta_train = pd.read_csv(f'{input_dir}/{dataset}/theta_{i}.csv')['theta'].values
            df_z_train = pd.read_csv(f'{input_dir}/{dataset}/z_train_{i}.csv')
            train_indices = df_z_train['index'].values
            z_train = df_z_train['z'].values
            df_z_test = pd.read_csv(f'{input_dir}/{dataset}/z_test_{i}.csv')
            test_indices = df_z_test['index'].values
            z_test = df_z_test['z'].values
            
            gof_mean, gof_std = goodness_of_fit_1PL(
                z=torch.tensor(z_hat, dtype=torch.float32),
                theta=torch.tensor(theta_hat, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.float32),
                plot_path=f"{plot_dir}/goodness_of_fit_{dataset}",
            )
            gof_means.append(gof_mean)
            gof_stds.append(gof_std)
            
            corr_ctt_mean, corr_ctt_std = theta_corr_ctt(
                theta=theta_hat,
                y=y,
                plot_path=f"{plot_dir}/theta_corr_ctt_{dataset}",
            )
            corr_ctt_means.append(corr_ctt_mean)
            corr_ctt_stds.append(corr_ctt_std)

    error_bar_plot(
        datasets=datasets,
        means=gof_means,
        stds=gof_stds,
        plot_path=f"{plot_dir}/summarize_goodness_of_fit",
    )
    
    error_bar_plot(
        datasets=datasets,
        means=corr_ctt_means,
        stds=corr_ctt_stds,
        plot_path=f"{plot_dir}/summarize_theta_corr_ctt",
    )
    
