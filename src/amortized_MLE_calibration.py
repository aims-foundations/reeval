import numpy as np
import torch
from tqdm import tqdm
from utils import item_response_fn_1PL, set_seed
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from MLE_calibration import MLE_calibration
from datasets import load_dataset

def amortized_MLE_calibration(response_matrix, embedding, device):
    # response_matrix [69, 959]; embedding [959, 4096]
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(response_matrix.size(0),), requires_grad=True, device=device)
    W = torch.normal(mean=0.0, std=0.01, size=(embedding.size(1),), requires_grad=True, device=device)
    
    optimizer = optim.Adam([W, theta_hat], lr=0.001, weight_decay=0.01)
    torch.nn.utils.clip_grad_norm_([W, theta_hat], max_norm=1.0)
    
    pbar = tqdm(range(3000))
    for _ in pbar:
        z3 = torch.matmul(embedding, W) # z3 [959]
        theta_hat_matrix = theta_hat.unsqueeze(1) # (n, 1)
        z3_matrix = z3.unsqueeze(0) # (1, m)
        
        prob_matrix = item_response_fn_1PL(z3_matrix, theta_hat_matrix)
        berns = torch.distributions.Bernoulli(prob_matrix.flatten())
        
        loss = -berns.log_prob(response_matrix.flatten()).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_postfix({'loss': loss.item()})

    return theta_hat, z3, W
    
if __name__ == "__main__":
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
    
    dataset = load_dataset("stair-lab/airbench-embedding", split="train")
    embeddings = dataset['embedding']
    emb_tensor = torch.tensor(embeddings).to(device) # [1199, 4096]
    
    assert response_matrix.shape[1] == emb_tensor.shape[0]
    
    response_matrix_train = response_matrix[:, :train_size]
    emb_train = emb_tensor[:train_size]
    emb_test = emb_tensor[train_size:]
    
    theta_amor_train, z3_amor_train, W_train = amortized_MLE_calibration(response_matrix_train, emb_train, device)
    z3_amor_train = z3_amor_train.cpu().detach().numpy()
    z3_amor_test = torch.matmul(emb_test, W_train)
    z3_amor_test = z3_amor_test.cpu().detach().numpy()
    
    assert z3_amor_train.shape == z3_nonamor_train.shape
    assert z3_amor_test.shape == z3_nonamor_test.shape
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(z3_nonamor_train, z3_amor_train, label='Train Z values')
    plt.xlabel('z3_nonamor_train')
    plt.ylabel('z3_amor_train')
    plt.title('Training: Non-amortized vs Amortized Z3')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(z3_nonamor_test, z3_amor_test, label='Test Z values')
    plt.xlabel('z3_nonamor_test')
    plt.ylabel('z3_amor_test')
    plt.title('Test: Non-amortized vs Amortized Z3')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../plot/real/amor_nonamor_train_test_comparison.png')

    corr_train = np.corrcoef(z3_nonamor_train, z3_amor_train)[0, 1]
    corr_test = np.corrcoef(z3_nonamor_test, z3_amor_test)[0, 1]
    print(f"Correlation between non-amortized and amortized Z3 values in training set: {corr_train}")
    print(f"Correlation between non-amortized and amortized Z3 values in test set: {corr_test}")