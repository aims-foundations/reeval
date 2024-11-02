import argparse
import os
import numpy as np
import torch
import wandb
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from utils import (
    set_seed, 
    DATASETS,
    plot_hist,
    item_response_fn_1PL_multi_dim, 
    goodness_of_fit_1PL_multi_dim_plot, 
    error_bar_plot_single,
    theta_corr_ctt,
    error_bar_plot_double,
)

def mle_multi_dim_amor_theta(
    response_matrix: torch.Tensor,
    constraint: bool,
    feat_matrix: torch.Tensor,
    dim: int=2,
    max_epoch: int=3000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response_matrix = response_matrix.to(device)
    num_model, num_item = response_matrix.shape
    feat_matrix = feat_matrix.to(device)

    W = torch.normal(
        mean=0.0, std=1.0,
        size=(feat_matrix.shape[1], dim),
        requires_grad=True, device=device
    )
    b = torch.normal(
        mean=0.0, std=1.0,
        size=(dim,),
        requires_grad=True, device=device
    )
    a = torch.normal(
        mean=0.0, std=1.0,
        size=(num_item, dim),
        requires_grad=True, device=device
    )
    z_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(num_item,),
        requires_grad=True, device=device
    )

    optimizer = optim.Adam([W, b, a, z_hat], lr=0.01)
    
    last_W = None
    last_b = None
    if constraint:
        last_a_softmax = None
    # else:
    #     last_a = None
    last_z_hat = None
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        if constraint:
            # b_full = b[None, :].repeat(num_model, 1) # (num_model, dim=2)
            # theta_hat = torch.mm(feat_matrix, W) + b_full # (num_model, dim=2)
            theta_hat = torch.mm(feat_matrix, W) + b # (num_model, dim=2)
            a_softmax = torch.nn.functional.softmax(a, dim=1)
            prob_matrix = item_response_fn_1PL_multi_dim(z_hat[None, :], theta_hat, a_softmax)
        # else:   
        #     prob_matrix = item_response_fn_1PL_multi_dim(z_hat[None, :], theta_hat, a)
        assert prob_matrix.shape == response_matrix.shape

        mask = response_matrix != -1
        masked_response_matrix = response_matrix[mask]
        masked_prob_matrix = prob_matrix[mask]

        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item()})
        # wandb.log({'loss': loss.item()})
        
        if constraint:
            if not (torch.isnan(W).any() or torch.isnan(b).any() or torch.isnan(a_softmax).any() or torch.isnan(z_hat).any()):
                last_W = W.cpu().detach().clone()
                last_b = b.cpu().detach().clone()
                last_a_softmax = a_softmax.cpu().detach().clone()
                last_z_hat = z_hat.cpu().detach().clone()
            else:
                break
        # else:
        #     if not (torch.isnan(theta_hat).any() or torch.isnan(a).any() or torch.isnan(z_hat).any()):
        #         last_theta_hat = theta_hat.cpu().detach().clone()
        #         last_a = a.cpu().detach().clone()
        #         last_z_hat = z_hat.cpu().detach().clone()
        #     else:
        #         break
    
    if constraint:
        return last_W, last_b, last_a_softmax, last_z_hat
    # else:
    #     return last_theta_hat, last_a, last_z_hat

if __name__ == "__main__":
    # wandb.init(project="mle_multi_dim_amor_theta")
    parser = argparse.ArgumentParser()
    parser.add_argument('--constraint', type=str, default='True', choices=['True', 'False'])
    args = parser.parse_args()
    
    if args.constraint == 'True':
        args.constraint = True
    # elif args.constraint == 'False':
    #     args.constraint = False
    
    set_seed(42)
    output_dir = f'../data/mle_multi_dim_amor_theta'
    plot_dir = f'../plot/mle_multi_dim_amor_theta'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    model_id_df = pd.read_csv('configs/model_id_ver1.csv')
    valid_model_names = model_id_df['model_names_reeval'].values
    feat_matrix = model_id_df[['Model Size (B)', 'Pretraining Data Size (T)', 'FLOPs (1E21)']].values
    feat_matrix = (feat_matrix - feat_matrix.mean(axis=0)) / feat_matrix.std(axis=0)
    split_index = int(len(valid_model_names) * 0.8)
    valid_model_names_train = valid_model_names[:split_index]
    feat_matrix_train = feat_matrix[:split_index]
    valid_model_names_test = valid_model_names[split_index:]
    feat_matrix_test = feat_matrix[split_index:]
    
    valid_datasets = []
    combined_matrix = pd.DataFrame()
    for dataset in DATASETS:
        matrix = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
        filtered_matrix = matrix[matrix.index.isin(valid_model_names_train)]
        # print(f"Dataset: {dataset}, left model num: {filtered_matrix.shape[0]}, left models: {filtered_matrix.index.tolist()}")
        print(f"Dataset: {dataset}, left model num: {filtered_matrix.shape[0]}")
        if not filtered_matrix.empty:
            valid_datasets.append(dataset)
            if combined_matrix.empty:
                combined_matrix = filtered_matrix
            else:
                combined_matrix = combined_matrix.join(filtered_matrix, how='outer', rsuffix='_dup')
    combined_matrix.fillna(-1, inplace=True)
    print(combined_matrix.shape)
    combined_matrix.to_csv(f"{output_dir}/combined_matrix.csv")
    # combined_matrix = pd.read_csv(f"{output_dir}/combined_matrix.csv", index_col=0)
    
    W, b, a, z_hat = mle_multi_dim_amor_theta(
        response_matrix=torch.tensor(combined_matrix.values, dtype=torch.float32),
        constraint=args.constraint,
        feat_matrix=torch.tensor(feat_matrix_train, dtype=torch.float32),
    )
    z_df = pd.DataFrame(z_hat.cpu().detach().numpy(), columns=["z"])
    z_df.to_csv(f"{output_dir}/z_con_{args.constraint}.csv", index=False)
    np.save(f"{output_dir}/W_con_{args.constraint}.npy", W.cpu().detach().numpy())
    np.save(f"{output_dir}/b_con_{args.constraint}.npy", b.cpu().detach().numpy())
    np.save(f"{output_dir}/a_con_{args.constraint}.npy", a.cpu().detach().numpy())
    
    W = W.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    
    # z_hat = pd.read_csv(f"{output_dir}/z_con_{args.constraint}.csv")
    
    theta_corr_trains = []
    theta_corr_tests = []
    for dataset in tqdm(valid_datasets):
        matrix = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
        matrix_train = matrix[matrix.index.isin(valid_model_names_train)]
        matrix_test = matrix[matrix.index.isin(valid_model_names_test)]
        
        train_indices = [np.where(valid_model_names_train == name)[0][0] for name in matrix_train.index]
        test_indices = [np.where(valid_model_names_test == name)[0][0] for name in matrix_test.index]
        
        feat_train = feat_matrix_train[train_indices]
        feat_test = feat_matrix_test[test_indices]
        
        theta_train = feat_train @ W + b
        theta_test = feat_test @ W + b
        
        theta_corr_train, _, _ = theta_corr_ctt(theta_train, matrix_train.values)
        theta_corr_test, _, _ = theta_corr_ctt(theta_test, matrix_test.values)
        theta_corr_trains.append(theta_corr_train)
        theta_corr_tests.append(theta_corr_test)
        
    error_bar_plot_double(
        datasets=valid_datasets, 
        means_train=theta_corr_trains, stds_train=[0] * len(theta_corr_trains), 
        means_test=theta_corr_test, stds_test=[0] * len(theta_corr_tests),
        plot_path=f"{plot_dir}/mle_multi_dim_amor_theta_summarize_theta_corr_ctt_con_{args.constraint}",
        xlabel=r"Theta Correlation",
        xlim_upper=1.1,
        plot_std=False,
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # gof_means, gof_stds = [], []
    # a_means = []
    # for dataset in tqdm(DATASETS):
    #     matrix = pd.read_csv(f'../data/pre_calibration/{dataset}/matrix.csv', index_col=0)
    #     response_matrix = combined_matrix.loc[matrix.index, matrix.columns].values
    #     row_indices = [combined_matrix.index.get_loc(i) for i in matrix.index]
    #     col_indices = [combined_matrix.columns.get_loc(i) for i in matrix.columns]
        
    #     theta_hat_subset = theta_hat[row_indices].cpu().detach()
    #     z_hat_subset = z_hat[col_indices].cpu().detach()
    #     a_subset = a[col_indices].cpu().detach()
        
    #     gof_mean, gof_std = goodness_of_fit_1PL_multi_dim_plot(
    #         z=z_hat_subset, 
    #         theta=theta_hat_subset,
    #         a=a_subset,
    #         y=torch.tensor(response_matrix, dtype=torch.float32),
    #         plot_path=f'{plot_dir}/goodness_of_fit_con_{args.constraint}_{dataset}.png',
    #     )
    #     gof_means.append(gof_mean)
    #     gof_stds.append(gof_std)
        
    #     if args.constraint:
    #         a_mean = a_subset[:, 0].numpy().mean()
    #         a_means.append(a_mean)
    #         # plot_hist(
    #         #     data=a_subset[:, 0].numpy(),
    #         #     plot_path=f'{plot_dir}/a_histogram_{dataset}.png',
    #         #     ylabel='Histiogram of a',
    #         # )
        
    # error_bar_plot_single(
    #     datasets=DATASETS,
    #     means=gof_means,
    #     stds=gof_stds,
    #     plot_path=f"{plot_dir}/mle_multi_dim_amor_theta_summarize_gof_con_{args.constraint}",
    #     xlabel=r"Goodness of Fit",
    # )
    
    # if args.constraint:
    #     error_bar_plot_single(
    #         datasets=DATASETS,
    #         means=a_means,
    #         stds=[0] * len(a_means),
    #         plot_path=f"{plot_dir}/mle_multi_dim_amor_theta_summarize_a_con_{args.constraint}",
    #         xlabel=r"Mean of $a$",
    #     )