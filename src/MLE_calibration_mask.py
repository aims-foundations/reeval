import torch
from tqdm import tqdm
from utils import item_response_fn_1PL, set_seed
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

def MLE_calibration_mask(response_matrix, device):
    theta_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(0),),
        requires_grad=True,
        device=device
    )
    z3 = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(1),),
        requires_grad=True,
        device=device
    )

    optimizer = optim.Adam([theta_hat, z3], lr=0.01)
    
    pbar = tqdm(range(3000))
    for _ in pbar:
        theta_hat_matrix = theta_hat.unsqueeze(1)
        z3_matrix = z3.unsqueeze(0)
        prob_matrix = item_response_fn_1PL(z3_matrix, theta_hat_matrix)
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item()})

    return theta_hat, z3

def main(
    y_df_path,
    z3_r_path,
    theta_r_path,
    save_z3_path,
    save_theta_path, 
    fig_path,
    device
):
    y_df = pd.read_csv(y_df_path, index_col=0)
    response_matrix = torch.tensor(y_df.values, dtype=torch.float32, device=device)
    
    theta_py, z3_py_whole = MLE_calibration_mask(response_matrix, device)
    theta_py = theta_py.cpu().detach().numpy()
    z3_py_whole = z3_py_whole.cpu().detach().numpy()
    
    z3_df = pd.DataFrame(z3_py_whole, columns=["z3"])
    z3_df.to_csv(save_z3_path, index=False)
    theta_df = pd.DataFrame(theta_py, columns=["theta"])
    theta_df.to_csv(save_theta_path, index=False)
    
    z3_r = pd.read_csv(z3_r_path)['z3']
    theta_r = pd.read_csv(theta_r_path)['F1']
    z3_py = z3_py_whole[:z3_r.shape[0]]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(z3_py, z3_r)
    plt.xlabel(r'Our $z_3$')
    plt.ylabel(r'mirt $z_3$')
    corr_np = np.corrcoef(z3_py, z3_r)[0, 1]
    plt.title(f'Correlation: {corr_np:.2f}')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    
    plt.subplot(1, 2, 2)
    plt.scatter(theta_py, theta_r)
    plt.xlabel(r'Our $\theta$')
    plt.ylabel(r'mirt $\theta$')
    corr_np = np.corrcoef(theta_py, theta_r)[0, 1]
    plt.title(f'Correlation: {corr_np:.2f}')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(
        '../data/real/response_matrix/normal_syn_reason/mask_matrix.csv', 
        '../data/real/irt_result/normal_syn_reason/Z/non_mask_1PL_Z_clean.csv',
        '../data/real/irt_result/normal_syn_reason/theta/non_mask_1PL_theta.csv', 
        '../data/real/irt_result/pyMLE_normal_syn_reason/Z/mask_1PL_Z.csv',
        '../data/real/irt_result/pyMLE_normal_syn_reason/theta/mask_1PL_theta.csv',
        '../plot/real/maskpy_unmaskr_calibration_comparison.png',
        device
    )