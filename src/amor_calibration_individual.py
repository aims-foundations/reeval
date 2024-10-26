import argparse
import os
import numpy as np
import wandb
import pandas as pd
import torch
from tqdm import tqdm
import torch.optim as optim
from datasets import load_dataset
from utils import set_seed, item_response_fn_1PL, split_indices

def amor_calibration(
    response_matrix: torch.Tensor, # response_matrix [69, 959]
    embedding: torch.Tensor, # embedding [959, 4096]
    W_init_std=5e-5,
    lr_theta=0.01,
    lr_W=5e-6,
    max_epoch=50000,
    patience=5000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    embedding = embedding.to(device)
    theta_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(0),),
        requires_grad=True,
        dtype=torch.float32,
        device=device
    )
    W = torch.normal(
        mean=0.0, std=W_init_std, 
        size=(embedding.size(1),),
        requires_grad=True,
        dtype=torch.float32,
        device=device
    )
    
    optimizer_theta = optim.Adam([theta_hat], lr=lr_theta)
    optimizer_W = optim.Adam([W], lr=lr_W)
    
    no_improvement_count = 0
    best_loss = float('inf')
    
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        z_hat = torch.matmul(embedding, W) # z_hat [959]
        theta_hat_matrix = theta_hat.unsqueeze(1) # (n, 1)
        z_hat_matrix = z_hat.unsqueeze(0) # (1, m)
        prob_matrix = item_response_fn_1PL(z_hat_matrix, theta_hat_matrix)
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]
        
        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer_theta.step()
        optimizer_W.step()
        optimizer_theta.zero_grad()
        optimizer_W.zero_grad()
        
        pbar.set_postfix({'loss': loss.item()})
        wandb.log({'train_loss': loss.item()})
        
        if abs(loss.item() - best_loss) < 1e-5:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            best_loss = loss.item()
        
        if no_improvement_count >= patience:
            break

    return theta_hat, z_hat, W

def main(
    train_indices, test_indices,
    emb_train, z_train_true, 
    emb_test, z_test_true,
    y_train,
    df_z_train_path, df_z_test_path,
    df_theta_path,
):
    theta_train, z_train, W_train = amor_calibration(
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(emb_train, dtype=torch.float32),
    )
    z_test = torch.matmul(emb_test, W_train.cpu().detach())
    
    df_z_train = pd.DataFrame({
        'index': train_indices,
        'z_true': z_train_true,
        'z_pred': z_train.cpu().detach().numpy(),
    })
    df_z_train.to_csv(df_z_train_path, index=False)
    
    df_z_test = pd.DataFrame({
        'index': test_indices,
        'z_true': z_test_true,
        'z_pred': z_test.cpu().detach().numpy(),
    })
    df_z_test.to_csv(df_z_test_path, index=False)
    
    df_theta = pd.DataFrame({
        'theta': theta_train.cpu().detach().numpy(),
    })
    df_theta.to_csv(df_theta_path, index=False)

if __name__ == "__main__":
    wandb.init(project="amor_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'])
    args = parser.parse_args()
    
    input_dir = '../data/pre_calibration/'
    output_dir = f'../data/amor_calibration/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed(args.seed)
    dataset = load_dataset(f'stair-lab/reeval_individual-embed', split=args.dataset)
    emb = np.array(dataset['embed'])
    z = np.array(dataset['z'])
    
    train_indices, test_indices = split_indices(z.shape[0])    
    emb_train, z_train_true = emb[train_indices], z[train_indices]
    emb_test, z_test_true = emb[test_indices], z[test_indices]

    y = pd.read_csv(f'{input_dir}/{args.dataset}/matrix.csv', index_col=0).values
    y_train = y[:, train_indices]

    main(
        train_indices, test_indices,
        emb_train, z_train_true, 
        emb_test, z_test_true,
        y_train,
        df_z_train_path=f'{output_dir}/z_train_{args.seed}.csv',
        df_z_test_path=f'{output_dir}/z_test_{args.seed}.csv',
        df_theta_path=f'{output_dir}/theta_{args.seed}.csv',
    )