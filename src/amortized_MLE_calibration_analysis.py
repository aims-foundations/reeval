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
    
def main(
    exp,
    y_df,
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
    elif args.exp == "mmlu":
        y_df = pd.read_csv('../data/real/response_matrix/normal_mmlu/non_mask_matrix.csv', index_col=0)
        theta_corr_path = '../data/real/irt_result/normal_mmlu/theta/pyMLE_mask_1PL_theta_manual.csv'
    elif args.exp == "syn_rea":
        y_df = pd.read_csv('../data/real/response_matrix/normal_syn_reason/mask_matrix.csv', index_col=0)
        theta_corr_path = '../data/real/irt_result/pyMLE_normal_syn_reason/theta/mask_1PL_theta_manual.csv'
    
    z3_nonamor_train, z3_nonamor_test, z3_amor_train, z3_amor_test,\
        theta_nonamor, theta_amor_train, losses= main(
            exp=args.exp,
            y_df=y_df,
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
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses during training')
    plt.savefig(f'../plot/real/{args.exp}_amor_losses.png')
    
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
    