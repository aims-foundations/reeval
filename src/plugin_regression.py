import argparse
import numpy as np
from datasets import load_dataset
import pandas as pd
from sklearn.linear_model import BayesianRidge
import pickle
import os

import torch
import wandb
from utils import bootstrap_mean_var, goodness_of_fit_1PL

def mse_with_var(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    se = np.square(y_true - y_pred)
    mean, var = bootstrap_mean_var(se)
    return mean, var

def plugin_regression(
    hf_repo,
    save_model_path,
):
    dataset_train = load_dataset(hf_repo, split="train")
    X_train = dataset_train['embed']
    y_train = dataset_train['z']
    X_train = np.array(X_train)
    print(f'Shape of X: {X_train.shape}') 
    y_train = np.array(y_train)
    
    dataset_test = load_dataset(hf_repo, split="test")
    X_test = dataset_test['embed']
    y_test = dataset_test['z']
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    model = BayesianRidge()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mean_mse_train, var_mse_train = mse_with_var(y_train, y_train_pred)
    print(f'train mse = {mean_mse_train}, var = {var_mse_train}')
    mean_mse_test, var_mse_test = mse_with_var(y_test, y_test_pred)
    print(f'test mse = {mean_mse_test}, var = {var_mse_test}')
        
    y_mean_pred = np.mean(y_train)
    mean_mse_base_train, var_mse_base_train = mse_with_var(y_train, np.full_like(y_train, y_mean_pred))
    print(f'train baseline mse = {mean_mse_base_train}, var = {var_mse_base_train}')
    mean_mse_base_test, var_mse_base_test = mse_with_var(y_test, np.full_like(y_test, y_mean_pred))
    print(f'test baseline mse = {mean_mse_base_test}, var = {var_mse_base_test}')
        
    with open(save_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return y_train_pred, y_test_pred
    
if __name__ == "__main__":
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    output_dir = f'../data/plugin_regression/{args.dataset}'
    plot_dir = f'../plot/plugin_regression'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    y_train_pred, y_test_pred = plugin_regression(
        hf_repo = f'stair-lab/reeval_{args.dataset}-embed',
        save_model_path=f'{output_dir}/bayridge.pkl',
    )

    theta = pd.read_csv(f'../data/calibration/{args.dataset}/nonamor_theta.csv')['theta'].values
    y = pd.read_csv(f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0).values
    split_index = int(0.8 * len(theta))
    theta_train, theta_test = theta[:split_index], theta[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    goodness_of_fit_1PL(
        z=torch.tensor(y_train_pred, dtype=torch.float32),
        theta=torch.tensor(theta, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        plot_path=f'{plot_dir}/goodness_of_fit_train_{args.dataset}.png',
    )
    
    goodness_of_fit_1PL(
        z=torch.tensor(y_test_pred, dtype=torch.float32),
        theta=torch.tensor(theta, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        plot_path=f'{plot_dir}/goodness_of_fit_test_{args.dataset}.png',
    )