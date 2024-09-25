import argparse
import numpy as np
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
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, concatenate_datasets

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

def train_model(
    model_name: str,
    emb_train: torch.Tensor, 
    emb_test: torch.Tensor,
    z_train: torch.Tensor, 
    z_test: torch.Tensor,
    batch_size: int=4096, 
    max_epoch: int=50, 
    lr: float=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = emb_train.shape[1]
    if model_name == 'mlp':
        model = MLP(input_dim).to(device)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(emb_train, z_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(emb_test, z_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        total_train_loss = 0
        model.train()
        for emb_batch, z_batch in train_loader:
            emb_batch, z_batch = emb_batch.to(device), z_batch.to(device)
            optimizer.zero_grad()
            outputs = model(emb_batch)
            loss = criterion(outputs, z_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        total_test_loss = 0
        model.eval()
        with torch.no_grad():
            for emb_batch, z_batch in test_loader:
                emb_batch = emb_batch.to(device)
                outputs = model(emb_batch)
                loss = criterion(outputs, z_batch)
                total_test_loss += loss.item()
            
        print(f'train_loss: {total_train_loss / len(train_loader)}, test_loss: {total_test_loss / len(test_loader)}')
        # pbar.set_postfix({'train_loss': total_train_loss / len(train_loader), 'test_loss': total_test_loss / len(test_loader)})

    model.eval()
    with torch.no_grad():
        z_train_pred = []
        for emb_batch in train_loader:
            emb_batch = emb_batch[0].to(device)
            outputs = model(emb_batch)
            z_train_pred.append(outputs.cpu().numpy())
        z_train_pred = np.concatenate(z_train_pred).flatten()

        z_test_pred = []
        for emb_batch in test_loader:
            emb_batch = emb_batch[0].to(device)
            outputs = model(emb_batch)
            z_test_pred.append(outputs.cpu().numpy())
        z_test_pred = np.concatenate(z_test_pred).flatten()
    
    return z_train_pred, z_test_pred, model.cpu()

def main(
    emb,
    z,
    model_name,
    df_train_path,
    df_test_path,
    save_model_path=None,
):
    train_indices, test_indices = split_indices(z.shape[0])    
    emb_train, z_train = emb[train_indices], z[train_indices]
    emb_test, z_test = emb[test_indices], z[test_indices]
    
    z_train_pred, z_test_pred, model = train_model(
        model_name=model_name,
        emb_train=torch.tensor(emb_train, dtype=torch.float32),
        emb_test=torch.tensor(emb_test, dtype=torch.float32),
        z_train=torch.tensor(z_train, dtype=torch.float32).view(-1, 1),
        z_test = torch.tensor(z_test, dtype=torch.float32).view(-1, 1),
    )
    
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
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'])
    args = parser.parse_args()
    
    output_dir = f'../data/plugin_regression/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    hf_repo=f'stair-lab/reeval_{args.dataset}-embed'
    dataset_train = load_dataset(hf_repo, split="train")
    dataset_test = load_dataset(hf_repo, split="test")
    dataset = concatenate_datasets([dataset_train, dataset_test])
    emb = np.array(dataset['embed'])
    z = np.array(dataset['z'])

    for i in tqdm(range(10)):
        set_seed(i)
        main(
            emb = emb,
            z = z,
            model_name=args.model,
            df_train_path=f'{output_dir}/train_{i}.csv',
            df_test_path=f'{output_dir}/test_{i}.csv',
            save_model_path=f'{output_dir}/{args.model}.pkl' if i==0 else None,
        )
        