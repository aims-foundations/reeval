
import argparse
from datasets import load_dataset
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
        theta = pd.read_csv('../data/real/irt_result/appendix1/theta/all_1PL_theta.csv')['F1'].to_numpy()
    # elif args.exp == "mmlu":
    
    save_path=f'../data/real/ppo/{args.exp}/{args.regression_model}_model.pkl',
    embed_repo=f'stair-lab/{args.exp}-embedding',

    with open('../../data/real/ppo/bayesian_ridge_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    emb_hf = load_dataset(embed_repo, split="whole")
    X = emb_hf['embeddings']
    Z = model.predict(X).tolist()
    
    goodness_of_fit_1PL(
        Z=Z,
        theta=theta,
        y_df=y_df,
        plot_path=f'../plot/real/{args.dataset}_pyMLE_goodness_of_fit.png',
        bin_size=7,
    )
   