
import argparse
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import pickle
from goodness_of_fit import goodness_of_fit_1PL
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--regression_model', type=str, default="bayridge")
    args = parser.parse_args()
    
    if args.exp == "airbench":
        y_df = pd.read_csv('../data/real/response_matrix/normal/all_matrix.csv', index_col=0)
        theta = pd.read_csv('../data/real/irt_result/normal/theta/all_1PL_theta.csv')['F1'].values
    elif args.exp == "mmlu":
        y_df = pd.read_csv('../data/real/response_matrix/normal_mmlu/non_mask_matrix.csv', index_col=0)
        theta = pd.read_csv('../data/real/irt_result/normal_mmlu/theta/pyMLE_mask_1PL_theta.csv')['theta'].values
    elif args.exp == "syn_rea":
        y_df = pd.read_csv('../data/real/response_matrix/normal_syn_reason/mask_matrix.csv', index_col=0)
        theta = pd.read_csv('../data/real/irt_result/pyMLE_normal_syn_reason/theta/mask_1PL_theta.csv')['theta'].values

    save_path=f'../data/real/ppo/{args.exp}/{args.regression_model}_model.pkl'
    embed_repo=f'stair-lab/{args.exp}-embedding'

    with open(save_path, 'rb') as f:
        model = pickle.load(f)
    
    emb_hf = load_dataset(embed_repo, split="whole")
    X = emb_hf['embeddings']
    Z = model.predict(X).tolist()
    
    train_size = int(len(Z) * 0.8)
    Z_train, Z_test = Z[:train_size], Z[train_size:]
    y_df_train, y_df_test = y_df.iloc[:, :train_size], y_df.iloc[:, train_size:]
    
    goodness_of_fit_1PL(
        Z=Z_train,
        theta=theta,
        y_df=y_df_train,
        plot_path=f'../plot/real/{args.exp}_plugin_goodness_of_fit_train.png',
        bin_size=7,
    )
    
    goodness_of_fit_1PL(
        Z=Z_test,
        theta=theta,
        y_df=y_df_test,
        plot_path=f'../plot/real/{args.exp}_plugin_goodness_of_fit_test.png',
        bin_size=7,
    )
   