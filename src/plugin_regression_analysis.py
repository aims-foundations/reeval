import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from sklearn.linear_model import BayesianRidge
import pickle
import os
import torch
from utils import bootstrap_mean_var, goodness_of_fit_1PL, set_seed

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
    # emb_train = dataset_train['embed']
    # z_train = dataset_train['z']
    # emb_train = np.array(emb_train)
    # print(f'Shape of X: {emb_train.shape}') 
    # z_train = np.array(z_train)
    
    dataset_test = load_dataset(hf_repo, split="test")
    # embed_test = dataset_test['embed']
    # z_test = dataset_test['z']
    # embed_test = np.array(embed_test)
    # z_test = np.array(z_test)
    
    dataset = concatenate_datasets([dataset_train, dataset_test])
    X = dataset['embed']
    y = dataset['z']

    X = np.array(X)
    print(f'Shape of X: {X.shape}') 
    y = np.array(y)
    
    emb_train, embed_test, z_train, z_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = BayesianRidge()
    model.fit(emb_train, z_train)

    z_train_pred = model.predict(emb_train)
    z_test_pred = model.predict(embed_test)
    mean_mse_train, var_mse_train = mse_with_var(z_train, z_train_pred)
    print(f'train mse = {mean_mse_train}, var = {var_mse_train}')
    mean_mse_test, var_mse_test = mse_with_var(z_test, z_test_pred)
    print(f'test mse = {mean_mse_test}, var = {var_mse_test}')
        
    z_mean_pred = np.mean(z_train)
    mean_mse_base_train, var_mse_base_train = mse_with_var(z_train, np.full_like(z_train, z_mean_pred))
    print(f'train baseline mse = {mean_mse_base_train}, var = {var_mse_base_train}')
    mean_mse_base_test, var_mse_base_test = mse_with_var(z_test, np.full_like(z_test, z_mean_pred))
    print(f'test baseline mse = {mean_mse_base_test}, var = {var_mse_base_test}')
        
    with open(save_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return z_train_pred, z_test_pred
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    output_dir = f'../data/plugin_regression/{args.dataset}'
    plot_dir = f'../plot/plugin_regression'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    z_train_pred, z_test_pred = plugin_regression(
        hf_repo = f'stair-lab/reeval_{args.dataset}-embed',
        save_model_path=f'{output_dir}/bayridge.pkl',
    )

    theta = pd.read_csv(f'../data/nonamor_calibration/{args.dataset}/nonamor_theta.csv')['theta'].values
    y = pd.read_csv(f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0).values
    split_index = int(0.8 * y.shape[1])
    y_train, y_test = y[:, :split_index], y[:, split_index:]
    
    goodness_of_fit_1PL(
        z=torch.tensor(z_train_pred, dtype=torch.float32),
        theta=torch.tensor(theta, dtype=torch.float32),
        y=torch.tensor(y_train, dtype=torch.float32),
        plot_path=f'{plot_dir}/goodness_of_fit_train_{args.dataset}.png',
    )
    
    goodness_of_fit_1PL(
        z=torch.tensor(z_test_pred, dtype=torch.float32),
        theta=torch.tensor(theta, dtype=torch.float32),
        y=torch.tensor(y_test, dtype=torch.float32),
        plot_path=f'{plot_dir}/goodness_of_fit_test_{args.dataset}.png',
    )