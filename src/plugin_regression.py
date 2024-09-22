import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from sklearn.linear_model import BayesianRidge
import pickle
import os
from tqdm import tqdm
import wandb
from utils import set_seed, split_indices

def main(
    hf_repo,
    df_train_path,
    df_test_path,
    save_model_path=None,
):
    dataset_train = load_dataset(hf_repo, split="train")
    dataset_test = load_dataset(hf_repo, split="test")
    dataset = concatenate_datasets([dataset_train, dataset_test])
    emb = np.array(dataset['embed'])
    z = np.array(dataset['z'])
    
    train_indices, test_indices = split_indices(z.shape[0])    
    emb_train = emb[train_indices]
    z_train = z[train_indices]
    emb_test = emb[test_indices]
    z_test = z[test_indices]
    
    model = BayesianRidge()
    model.fit(emb_train, z_train)

    z_train_pred = model.predict(emb_train)
    z_test_pred = model.predict(emb_test)
    
    df_train = pd.DataFrame({
        'index': train_indices,
        'z_true': z_train,
        'z_pred': z_train_pred,
    })
    df_train.to_csv(df_train_path, index=False)
    
    df_test = pd.DataFrame({
        'index': test_indices,
        'z_true': z_test,
        'z_pred': z_test_pred,
    })
    df_test.to_csv(df_test_path, index=False)
    
    if save_model_path is not None:
        with open(save_model_path, 'wb') as f:
            pickle.dump(model, f)
    
if __name__ == "__main__":
    wandb.init(project="plugin_regression")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    output_dir = f'../data/plugin_regression/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(100)):
        set_seed(i)
        main(
            hf_repo=f'stair-lab/reeval_{args.dataset}-embed',
            df_train_path=f'{output_dir}/train_{i}.csv',
            df_test_path=f'{output_dir}/test_{i}.csv',
            save_model_path=f'{output_dir}/bayridge.pkl' if i==42 else None,
        )
        

    
    