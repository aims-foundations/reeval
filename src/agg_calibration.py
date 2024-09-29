import argparse
import json
import warnings
import os
import pandas as pd
import torch
from tqdm import tqdm
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from utils import (
    set_seed, 
    item_response_fn_1PL, 
    split_indices, 
    DATASETS, 
    MLP
)

class BatchDataset(Dataset):
    def __init__(self, emb, y):
        self.emb = emb
        self.y = y
    def __len__(self):
        return self.emb.shape[0]
    def __getitem__(self, idx):
        emb = self.emb[idx, :]
        y = self.y[:, idx]
        return emb, y
    
def agg_amor_calibration(
    datasets: list[str], 
    train_indices: list[list[int]],
    test_indices: list[list[int]],
    emb_hf_repo: str,
    model_id_path: str,
    lr_theta=0.01,
    lr_mlp=1e-3,
    max_epoch=10,
    embed_dim=4096,
    bs=4096,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(model_id_path, 'r') as f:
        model_id_dict = json.load(f)
    
    mlp_model = MLP(embed_dim).to(device)
    mlp_model.train()
    theta_train = torch.normal(
        mean=0.0, std=1.0,
        size=(len(model_id_dict),),
        requires_grad=True,
        dtype=torch.float32,
        device=device
    )
            
    optimizer_theta = optim.Adam([theta_train], lr=lr_theta)
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=lr_mlp)
            
    z_trains = []
    for epoch in tqdm(range(max_epoch), desc='Training'):
        pbar = tqdm(datasets, desc='Dataset')
        for i, dataset in enumerate(pbar):
            train_index = train_indices[i]
            
            y_df = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
            y = torch.tensor(y_df.values[:, train_index]).to(device)
            
            model_names = y_df.index.tolist()
            model_ids = [model_id_dict[name] for name in model_names]
            theta_train_subset = theta_train[model_ids]
            
            hf_repo = load_dataset(emb_hf_repo, split=dataset)
            emb = torch.tensor(hf_repo['embed'])[train_index].to(device)
            
            assert y.shape[0] == theta_train_subset.shape[0]
            assert y.shape[1] == emb.shape[0]
            
            dataset_batch = BatchDataset(emb, y)
            data_loader = DataLoader(dataset_batch, batch_size=bs, shuffle=True)

            for emb_batch, y_batch in data_loader:
                y_batch = y_batch.T
                z_train = mlp_model(emb_batch).flatten()
                theta_train_matrix = theta_train_subset.unsqueeze(1)  # (n, 1)
                z_train_matrix = z_train.unsqueeze(0)  # (1, m)
                prob_matrix = item_response_fn_1PL(z_train_matrix, theta_train_matrix)
                assert prob_matrix.shape == y_batch.shape
                
                mask = y_batch!=-1
                masked_y_batch = y_batch.flatten()[mask.flatten()]
                masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]
                
                berns = torch.distributions.Bernoulli(masked_prob_matrix)
                loss = -berns.log_prob(masked_y_batch).mean()
                loss.backward()
                optimizer_theta.step()
                optimizer_mlp.step()
                optimizer_theta.zero_grad()
                optimizer_mlp.zero_grad()
                
                pbar.set_postfix({'loss': loss.item()})
            
            if epoch == max_epoch-1:
                z_trains.append(z_train)
    
    z_tests = []
    for i, dataset in enumerate(tqdm(datasets, desc='Testing')):
        test_index = test_indices[i]
        
        y_df = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
        y = torch.tensor(y_df.values[:, test_index]).to(device)
        
        hf_repo = load_dataset(emb_hf_repo, split=dataset)
        emb = torch.tensor(hf_repo['embed'])[test_index].to(device)
        
        z_test = mlp_model(emb).flatten()
        z_tests.append(z_test)
    
    return theta_train, z_trains, z_tests

def main(
    datasets,
    emb_hf_repo,
    model_id_path,
    iteration,
):
    train_indices, test_indices = [], []
    for dataset in datasets:
        y = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
        train_index, test_index = split_indices(y.shape[1])
        train_indices.append(train_index)
        test_indices.append(test_index)
    
    theta_train, z_trains, z_tests = agg_amor_calibration(
        datasets=datasets, 
        train_indices=train_indices,
        test_indices=test_indices,
        emb_hf_repo=emb_hf_repo,
        model_id_path=model_id_path,
    )
    
    for i, dataset in enumerate(tqdm(datasets, desc='Saving')):
        output_dir = f'../data/agg_calibration/{dataset}'
        os.makedirs(output_dir, exist_ok=True)
        df_z_train_path=f'{output_dir}/z_train_{iteration}.csv'
        df_z_test_path=f'{output_dir}/z_test_{iteration}.csv'
        
        df_z_train = pd.DataFrame({
            'index': train_indices[i],
            'z': z_trains[i].cpu().detach().numpy(),
        })
        df_z_train.to_csv(df_z_train_path, index=False)
        
        df_z_test = pd.DataFrame({
            'index': test_indices[i],
            'z': z_tests[i].cpu().detach().numpy(),
        })
        df_z_test.to_csv(df_z_test_path, index=False)
        
    df_theta_path=f'../data/agg_calibration/theta_{iteration}.csv'
    df_theta = pd.DataFrame({
        'theta': theta_train.cpu().detach().numpy()
    })
    df_theta.to_csv(df_theta_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'])
    parser.add_argument('--task', type=str, default='byrandom', choices=['byrandom', 'bydataset'])
    args = parser.parse_args()
    
    for i in tqdm(range(10), desc='Seed'):
        set_seed(i)
        main(
            datasets=DATASETS,
            emb_hf_repo=f'stair-lab/reeval_aggregate-embed',
            model_id_path='configs/model_id.json',
            iteration=i,
        )
        