import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from utils import set_seed, split_indices
import ast

class RidgeRegression(nn.Module):
    def __init__(self, input_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_ridge_model(
    emb_train, 
    z_train, 
    emb_test, 
    l2_reg=1, 
    max_epoch=5000, 
    lr=0.1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = len(emb_train)
    model = RidgeRegression(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=l2_reg
    )

    emb_train_tensor = torch.tensor(
        emb_train, dtype=torch.float32, device=device
    )
    z_train_tensor = torch.tensor(
        z_train, dtype=torch.float32, device=device
    ).view(-1, 1)
    emb_test_tensor = torch.tensor(
        emb_test, dtype=torch.float32, device=device
    )

    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        model.train()
        optimizer.zero_grad()
        outputs = model(emb_train_tensor)
        loss = criterion(outputs, z_train_tensor)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
    
    model.eval()
    with torch.no_grad():
        z_train_pred = model(emb_train_tensor).cpu().detach().numpy().flatten()
        z_test_pred = model(emb_test_tensor).cpu().detach().numpy().flatten()
        
    return z_train_pred, z_test_pred, model.cpu()

def main(
    hf_repo,
    df_train_path,
    df_test_path,
    save_model_path=None,
):
    # dataset_train = load_dataset(hf_repo, split="train")
    # dataset_test = load_dataset(hf_repo, split="test")
    # dataset = concatenate_datasets([dataset_train, dataset_test])
    dataset = load_dataset(hf_repo, split="train")
    emb = np.array(ast.literal_eval(dataset['embed']))
    z = np.array(dataset['z']) 
    
    train_indices, test_indices = split_indices(z.shape[0])    
    emb_train, z_train = emb[train_indices], z[train_indices]
    emb_test, z_test = emb[test_indices], z[test_indices]
    
    z_train_pred, z_test_pred, model = train_ridge_model(
        emb_train, z_train, emb_test
    )
    
    trian_mse = np.mean((z_train - z_train_pred.flatten())**2)
    test_mse = np.mean((z_test - z_test_pred.flatten())**2)
    print(f'Train MSE: {trian_mse}, Test MSE: {test_mse}')
    
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
    
    for i in tqdm(range(10)):
        set_seed(i)
        main(
            hf_repo=f'stair-lab/reeval_{args.dataset}-embed',
            df_train_path=f'{output_dir}/train_{i}.csv',
            df_test_path=f'{output_dir}/test_{i}.csv',
            save_model_path=f'{output_dir}/bayridge.pkl' if i==0 else None,
        )
        