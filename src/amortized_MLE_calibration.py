import argparse
import numpy as np
import torch
from tqdm import tqdm
import wandb
from utils import item_response_fn_1PL, set_seed
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from MLE_calibration import MLE_calibration
from datasets import load_dataset

def amortized_MLE_calibration(
    response_matrix,
    embedding, 
    device,
    lr_theta=0.01,
    W_init_std=5e-5,
    lr_W=5e-6,
    # weight_decay=1, 
    epochs=20000,
    # max_norm=1.0,
    # lr_decay_factor=0.8,  # Learning rate decay factor
    # lr_decay_step=50,    # Decay every 100 epochs
):
    # response_matrix [69, 959]; embedding [959, 4096]
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(response_matrix.size(0),), requires_grad=True, device=device)
    W = torch.normal(mean=0.0, std=W_init_std, size=(embedding.size(1),), requires_grad=True, device=device)
    
    # optimizer = optim.Adam([W, theta_hat], lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)
    # torch.nn.utils.clip_grad_norm_([W, theta_hat], max_norm=max_norm)
    
    optimizer_theta = optim.Adam([theta_hat], lr=lr_theta)
    optimizer_W = optim.Adam([W], lr=lr_W)
    
    losses = []
    pbar = tqdm(range(epochs))
    for _ in pbar:
        z3 = torch.matmul(embedding, W) # z3 [959]
        theta_hat_matrix = theta_hat.unsqueeze(1) # (n, 1)
        z3_matrix = z3.unsqueeze(0) # (1, m)
        prob_matrix = item_response_fn_1PL(z3_matrix, theta_hat_matrix)
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]
        
        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer_theta.step()
        optimizer_W.step()
        # scheduler.step()
        optimizer_theta.zero_grad()
        optimizer_W.zero_grad()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': loss.item()})

    return theta_hat, z3, W, losses
    
def main(
    lr,
    weight_decay,
    epochs,
    W_init_std,
    max_norm
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_df = pd.read_csv('../data/real/response_matrix/normal/all_matrix.csv', index_col=0)
    response_matrix = torch.tensor(y_df.values, dtype=torch.float32, device=device) # [69, 1199]
    train_size = int(0.8 * response_matrix.shape[1])
    
    theta_nonamor, z3_nonamor = MLE_calibration(response_matrix, device)
    z3_nonamor_train = z3_nonamor[:train_size]
    z3_nonamor_train = z3_nonamor_train.cpu().detach().numpy()
    z3_nonamor_test = z3_nonamor[train_size:]
    z3_nonamor_test = z3_nonamor_test.cpu().detach().numpy()
    
    dataset = load_dataset("stair-lab/airbench-embedding", split="whole")
    embeddings = dataset['embeddings']
    emb_tensor = torch.tensor(embeddings).to(device) # [1199, 4096]
    
    assert response_matrix.shape[1] == emb_tensor.shape[0]
    
    response_matrix_train = response_matrix[:, :train_size]
    emb_train = emb_tensor[:train_size]
    emb_test = emb_tensor[train_size:]
    
    theta_amor_train, z3_amor_train, W_train, _ = amortized_MLE_calibration(
        response_matrix_train,
        emb_train,
        device,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        W_init_std=W_init_std,
        max_norm=max_norm
    )
    z3_amor_train = z3_amor_train.cpu().detach().numpy()
    z3_amor_test = torch.matmul(emb_test, W_train)
    z3_amor_test = z3_amor_test.cpu().detach().numpy()
    
    test_mse = np.mean((z3_nonamor_test - z3_amor_test) ** 2)
    wandb.log({"test_mse": test_mse})
    
    return z3_nonamor_train, z3_nonamor_test, z3_amor_train, z3_amor_test
    
   
if __name__ == "__main__":
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--W_init_std", type=float)
    parser.add_argument("--max_norm", type=float)
    args = parser.parse_args()
    
    # z3_nonamor_train, z3_nonamor_test, z3_amor_train, z3_amor_test = main(
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     epochs=args.epochs,
    #     W_init_std=args.W_init_std,
    #     max_norm=args.max_norm
    # )
    
    # assert z3_amor_train.shape == z3_nonamor_train.shape
    # assert z3_amor_test.shape == z3_nonamor_test.shape
    
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.scatter(z3_nonamor_train, z3_amor_train, label='Train Z values')
    # plt.xlabel('z3_nonamor_train')
    # plt.ylabel('z3_amor_train')
    # plt.title('Training: Non-amortized vs Amortized Z3')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.scatter(z3_nonamor_test, z3_amor_test, label='Test Z values')
    # plt.xlabel('z3_nonamor_test')
    # plt.ylabel('z3_amor_test')
    # plt.title('Test: Non-amortized vs Amortized Z3')
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig('../plot/real/amor_nonamor_train_test_comparison.png')

    # corr_train = np.corrcoef(z3_nonamor_train, z3_amor_train)[0, 1]
    # corr_test = np.corrcoef(z3_nonamor_test, z3_amor_test)[0, 1]
    # print(f"Correlation between non-amortized and amortized Z3 values in training set: {corr_train}")
    # print(f"Correlation between non-amortized and amortized Z3 values in test set: {corr_test}")
    
    # mse_train = np.mean((z3_nonamor_train - z3_amor_train) ** 2)
    # mse_test = np.mean((z3_nonamor_test - z3_amor_test) ** 2)
    # print(f"MSE between non-amortized and amortized Z3 values in training set: {mse_train}")
    # print(f"MSE between non-amortized and amortized Z3 values in test set: {mse_test}")
    