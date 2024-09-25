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
from sklearn.metrics import mean_squared_error

class RidgeRegression(nn.Module):
    def __init__(self, input_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, 2048),
            nn.ELU(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_ridge_model(
    model_name,
    emb_train, 
    z_train, 
    emb_test, 
    max_epoch=5000, 
    lr=0.1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = emb_train.shape[1]
    if model_name == 'ridge':
        model = RidgeRegression(input_dim).to(device)
    elif model_name == 'mlp':
        model = MLP(input_dim).to(device)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
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
    dataset_train = load_dataset(hf_repo, split="train")
    dataset_test = load_dataset(hf_repo, split="test")
    dataset = concatenate_datasets([dataset_train, dataset_test])
    emb = np.array(dataset['embed'])
    z = np.array(dataset['z'])
    
    train_indices, test_indices = split_indices(z.shape[0])    
    emb_train, z_train = emb[train_indices], z[train_indices]
    emb_test, z_test = emb[test_indices], z[test_indices]
    
    z_train_pred, z_test_pred, model = train_ridge_model(
        emb_train, z_train, emb_test
    )
    
    # mse
    mse_train = mean_squared_error(z_train, z_train_pred)
    mse_test = mean_squared_error(z_test, z_test_pred)
    print(f'MSE Train: {mse_train:.2f}, MSE Test: {mse_test:.2f}')
    
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
    parser.add_argument('--model', type=str, default='mlp', choices=['ridge', 'mlp'])
    args = parser.parse_args()
    
    output_dir = f'../data/plugin_regression/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(10)):
        set_seed(i)
        main(
            model_name=args.model,
            hf_repo=f'stair-lab/reeval_{args.dataset}-embed',
            df_train_path=f'{output_dir}/train_{i}.csv',
            df_test_path=f'{output_dir}/test_{i}.csv',
            save_model_path=f'{output_dir}/{args.dataset}.pkl' if i==0 else None,
        )
        