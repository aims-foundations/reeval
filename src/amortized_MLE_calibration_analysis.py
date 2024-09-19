import argparse
import numpy as np
import torch
from utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')
from datasets import load_dataset
from amortized_MLE_calibration import amortized_MLE_calibration
from MLE_calibration_mask import MLE_calibration_mask
from goodness_of_fit import goodness_of_fit_1PL
from helm_theta_correlation import theta_corr_plot

def z_corr_plot(
    x,
    y,
    plot_path,
):
    corr = np.corrcoef(x, y)[0, 1]
    mse = np.mean((x - y) ** 2)
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.xlabel(r'$z$ from amortized IRT calibration', fontsize=45)
    plt.ylabel(r'$z$ from non-amortized IRT calibration', fontsize=45)
    plt.title(f'Correlation: {corr:.2f}, MSE: {mse:.2f}', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

def main(
    exp,
    y_df,
    epochs=20000
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = torch.tensor(y_df.values, dtype=torch.float32, device=device)
    train_size = int(0.8 * response_matrix.shape[1])
    
    theta_nonamor, z3_nonamor = MLE_calibration_mask(response_matrix, device)
    z3_nonamor_train = z3_nonamor[:train_size]
    z3_nonamor_train = z3_nonamor_train.cpu().detach().numpy()
    z3_nonamor_test = z3_nonamor[train_size:]
    z3_nonamor_test = z3_nonamor_test.cpu().detach().numpy()
    
    dataset = load_dataset(f"stair-lab/{exp}-embedding", split="whole")
    embeddings = dataset['embeddings']
    emb_tensor = torch.tensor(embeddings).to(device)
    
    assert response_matrix.shape[1] == emb_tensor.shape[0]
    
    response_matrix_train = response_matrix[:, :train_size]
    emb_train = emb_tensor[:train_size]
    emb_test = emb_tensor[train_size:]
    
    theta_amor_train, z3_amor_train, W_train, losses = amortized_MLE_calibration(
        response_matrix_train,
        emb_train,
        device,
        epochs=epochs
    )
    
    z3_amor_train = z3_amor_train.cpu().detach().numpy()
    z3_amor_test = torch.matmul(emb_test, W_train)
    z3_amor_test = z3_amor_test.cpu().detach().numpy()
    
    theta_nonamor = theta_nonamor.cpu().detach().numpy()
    theta_amor_train = theta_amor_train.cpu().detach().numpy()
    
    return z3_nonamor_train, z3_nonamor_test, z3_amor_train, z3_amor_test, \
        theta_nonamor, theta_amor_train, losses
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    args = parser.parse_args()
    
    if args.exp == "airbench":
        y_df = pd.read_csv('../data/real/response_matrix/normal/all_matrix.csv', index_col=0)
        theta_corr_path = '../data/real/irt_result/normal/theta/all_1PL_theta_manual.csv'
        epochs=20000
    elif args.exp == "mmlu":
        y_df = pd.read_csv('../data/real/response_matrix/normal_mmlu/non_mask_matrix.csv', index_col=0)
        theta_corr_path = '../data/real/irt_result/normal_mmlu/theta/pyMLE_mask_1PL_theta_manual.csv'
        epochs=50000
    elif args.exp == "syn_rea":
        y_df = pd.read_csv('../data/real/response_matrix/normal_syn_reason/mask_matrix.csv', index_col=0)
        theta_corr_path = '../data/real/irt_result/pyMLE_normal_syn_reason/theta/mask_1PL_theta_manual.csv'
        epochs=50000
    
    z3_nonamor_train, z3_nonamor_test, z3_amor_train, z3_amor_test,\
        theta_nonamor, theta_amor_train, losses= main(
            exp=args.exp,
            y_df=y_df,
            epochs=epochs,
    )
        
    y_df_train = y_df.iloc[:, :z3_nonamor_train.shape[0]]
    y_df_test = y_df.iloc[:, z3_nonamor_train.shape[0]:]
    
    assert z3_amor_train.shape[0] == z3_nonamor_train.shape[0] == y_df_train.shape[1]
    assert z3_amor_test.shape[0] == z3_nonamor_test.shape[0] == y_df_test.shape[1]
    
    goodness_of_fit_1PL(
        Z=z3_amor_train,
        theta=theta_amor_train,
        y_df=y_df_train,
        plot_path=f'../plot/real/{args.exp}_amor_goodness_of_fit_train.png',
    )
    
    goodness_of_fit_1PL(
        Z=z3_amor_test,
        theta=theta_amor_train,
        y_df=y_df_test,
        plot_path=f'../plot/real/{args.exp}_amor_goodness_of_fit_test.png',
    )
    
    df = pd.read_csv(theta_corr_path)
    y = df.loc[:, "score"].to_numpy()
    theta_corr_plot(
        x=theta_amor_train,
        y=y,
        plot_path=f'../plot/real/{args.exp}_amor_theta_corr.png',
    )
    
    plt.figure(figsize=(10, 10))
    plt.plot(losses)
    plt.xlabel(r'Epochs', fontsize=45)
    plt.ylabel(r'Loss', fontsize=45)
    plt.tick_params(axis='both', labelsize=35)
    plt.savefig(f'../plot/real/{args.exp}_amor_losses.png', dpi=300, bbox_inches='tight')
    
    z_corr_plot(
        x=z3_amor_train,
        y=z3_nonamor_train,
        plot_path=f'../plot/real/{args.exp}_amor_z_corr_train.png',
    )
    
    z_corr_plot(
        x=z3_amor_test,
        y=z3_nonamor_test,
        plot_path=f'../plot/real/{args.exp}_amor_z_corr_test.png',
    )
    