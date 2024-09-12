import torch
from tqdm import tqdm
from utils import item_response_fn_1PL, set_seed
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

def MLE_calibration(response_matrix, device):
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(response_matrix.size(0),), requires_grad=True, device=device)
    z3 = torch.normal(mean=0.0, std=1.0, size=(response_matrix.size(1),), requires_grad=True, device=device)

    optimizer = optim.Adam([theta_hat, z3], lr=0.01)
    
    pbar = tqdm(range(1000))
    for _ in pbar:
        theta_hat_matrix = theta_hat.unsqueeze(1)
        z3_matrix = z3.unsqueeze(0)
        prob_matrix = item_response_fn_1PL(z3_matrix, theta_hat_matrix)
        berns = torch.distributions.Bernoulli(prob_matrix.flatten())
        loss = -berns.log_prob(response_matrix.flatten()).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item()})

    return theta_hat, z3

def main(y_df_path, z3_r_path, theta_r_path, fig_path, device):
    y_df = pd.read_csv(y_df_path, index_col=0)
    response_matrix = torch.tensor(y_df.values, dtype=torch.float32, device=device)
    
    theta_py, z3_py = MLE_calibration(response_matrix, device)
    theta_py = theta_py.cpu().detach().numpy()
    z3_py = z3_py.cpu().detach().numpy()
    
    z3_r = pd.read_csv(z3_r_path)['z3']
    theta_r = pd.read_csv(theta_r_path)['F1']

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(z3_py, z3_r, label='Z values')
    plt.xlabel('Py z3')
    plt.ylabel('R z3')
    plt.title('Comparison of z3 values')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(theta_py, theta_r, label='Theta values')
    plt.xlabel('Py theta')
    plt.ylabel('R theta')
    plt.title('Comparison of Theta values')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_path)
    
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # synthetic
    main(
        '../data/synthetic/response_matrix/synthetic_matrix_1PL.csv', 
        '../data/synthetic/irt_result/R_mirt/Z/synthetic_1PL_Z_clean.csv', 
        '../data/synthetic/irt_result/R_mirt/theta/synthetic_1PL_theta.csv', 
        '../plot/synthetic/py_r_calibration_comparison.png',
        device
    )
    
    # real
    main(
        '../data/real/response_matrix/normal/all_matrix.csv', 
        '../data/real/irt_result/normal/Z/all_1PL_Z_clean.csv', 
        '../data/real/irt_result/normal/theta/all_1PL_theta.csv', 
        '../plot/real/py_r_calibration_comparison.png',
        device
    )