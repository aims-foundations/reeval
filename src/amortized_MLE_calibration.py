import torch
from tqdm import tqdm
from utils import item_response_fn_1PL, set_seed
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from embed_text_package.embed_text import Embedder
from MLE_calibration import MLE_calibration

def amortized_MLE_calibration(response_matrix, embedding, device):
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(response_matrix.size(0),), requires_grad=True, device=device)
    W = torch.normal(mean=0.0, std=1.0, size=(embedding.size(1),), requires_grad=True, device=device)
    
    optimizer = optim.Adam([W, theta_hat], lr=0.01, weight_decay=0.01)
    
    pbar = tqdm(range(1000))
    for _ in pbar:
        theta_hat_matrix = theta_hat.unsqueeze(1) # (n, 1)
        z3 = torch.matmul(embedding, W)
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
    
    y_df = pd.read_csv('../data/synthetic/response_matrix/synthetic_matrix_1PL.csv', index_col=0)
    response_matrix = torch.tensor(y_df.values, dtype=torch.float32, device=device)
    
    theta_nonamor, z3_nonamor = MLE_calibration(response_matrix, device)
    z3_nonamor_test = z3_nonamor[train_size:]
    z3_nonamor_test = z3_nonamor_test.cpu().detach().numpy()
    
    dataset = load_dataset("stair-lab/airbench-difficulty", split="whole")
    cols_to_be_embded = ['question_text']
    bs = 1024
    model_name = "meta-llama/Meta-Llama-3-8B"
    
    embdr = Embedder()
    embdr.load(model_name)
    dataloader = DataLoader(dataset, batch_size=bs)
    emb = embdr.get_embeddings(
        dataloader, model_name, cols_to_be_embded
    )
    
    assert response_matrix.shape[0] == emb.shape[0]
    
    train_size = int(0.8 * response_matrix.shape[0])
    response_matrix_train = response_matrix[:train_size]
    emb_train = emb[:train_size]
    emb_test = emb[train_size:]
    
    theta_amor_train, z3_amor_train, W_train = amortized_MLE_calibration(response_matrix_train, emb_train, device)
    
    z3_amor_test = torch.matmul(emb_test, W_train)
    z3_amor_test = z3_amor_test.cpu().detach().numpy()
    
    plt.figure(figsize=(5, 5))
    plt.scatter(z3_nonamor_test, z3_amor_test, label='Z values')
    plt.xlabel('z3_nonamor_test')
    plt.ylabel('z3_amor_test')
    plt.title('Comparison of z3 values')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../plot/synthetic/amor_nonamor_comparison.png')
    
    
    