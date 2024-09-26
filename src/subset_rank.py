import argparse
import os
import numpy as np
import pandas as pd
from torch import nn, optim
import torch
import wandb
from nonamor_calibration import nonamor_calibration
from utils import (
    set_seed,
    perform_t_test,
    sample_mean_std, 
    item_response_fn_1PL,
    get_theta_ctt
)
from tqdm import tqdm
from matplotlib import pyplot as plt

def fit_theta(asked_z, y, epoch=300):
    non_missing_indicators = y != -1    
    theta_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(y.shape[0],),
        requires_grad=True,
        device=y.device,
    )
    optimizer = optim.Adam([theta_hat], lr=0.01)
    epoch = 100
    pbar = tqdm(range(epoch))
    for _ in pbar:
        prob_matrix = item_response_fn_1PL(asked_z[None, :], theta_hat[:, None])
        prob_matrix = prob_matrix.flatten()[non_missing_indicators.flatten()]
        berns = torch.distributions.Bernoulli(prob_matrix)
        loss = -berns.log_prob(y.flatten()[non_missing_indicators.flatten()]).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pbar.set_postfix({'loss': loss.item()})

def main(
    # theta_path, 
    y_path,
    # plot_path,
    # output_path
):
    y = pd.read_csv(y_path, index_col=0).values
    non_missing_indicators = y != -1
    
    ctt_true_score = (y * non_missing_indicators).sum(1)/non_missing_indicators.sum(1)    
    ctt_true_ranks = np.argsort(ctt_true_score)
    
    subset_indices = np.random.choice(ctt_true_score.shape[0], size=15, replace=False)
    ctt_true_score_sub = ctt_true_score[subset_indices]

    # plot the ctt_true_score as vertical line 
    plt.figure(figsize=(10, 5))
    plt.scatter(ctt_true_score, np.ones_like(ctt_true_score), c='blue')
    plt.scatter(ctt_true_score_sub, np.ones_like(ctt_true_score_sub), c='red')    
    plt.show()
    
    question_num = y.shape[1]
    subset_size = min(y.shape[1], 500)
    print(f"subset size = {subset_size}")
    
    ctt_correctly_ranked = 0
    irt_correctly_ranked = 0
    n_simulation = 1000
    for _ in range(n_simulation):
        subset_indices = np.random.choice(question_num, size=subset_size, replace=False)
        y_sub = y[:, subset_indices]
        non_missing_indicators_sub = non_missing_indicators[:, subset_indices]
        
        ctt_sub = (y_sub * non_missing_indicators_sub).sum(1)/non_missing_indicators_sub.sum(1)
        ctt_sub_ranks = np.argsort(ctt_sub)
        
        irt_sub = nonamor_calibration(torch.tensor(y_sub, dtype=torch.float32))
        irt_sub_ranks = np.argsort(irt_sub)
        
        # check if the rank of the ctt and irt subset is the same as the rank of the whole dataset
        if np.all(ctt_sub_ranks == ctt_true_ranks):
            ctt_correctly_ranked += 1
            
        if np.all(irt_sub_ranks == ctt_true_ranks):
            irt_correctly_ranked += 1
            
    print(f"ctt_correctly_ranked = {ctt_correctly_ranked/n_simulation}")
    print(f"irt_correctly_ranked = {irt_correctly_ranked/n_simulation}")
    
    
        
        
        
    #     theta_new, _ = nonamor_calibration(torch.tensor(y_new, dtype=torch.float32))
    #     theta_new_masked, ctt_new_masked = get_theta_ctt(theta_new.cpu().detach().numpy(), y_new)
    #     theta_estimate = np.mean(theta_new_masked)
    #     ctt_estimate = np.mean(ctt_new_masked)
    #     theta_estimates.append(theta_estimate)
    #     ctt_estimates.append(ctt_estimate)
        
    
    # dumb_answers = y[i][easy_indices]
    # smart_answers = y[j][hard_indices]
    
    # # CTT
    # print("CTT")
    # mean_dumb, std_dumb = sample_mean_std(dumb_answers)
    # print(f"dumb CTT mean = {mean_dumb}")
    # print(f"dumb CTT std= {std_dumb}")
    
    # mean_smart, std_smart = sample_mean_std(smart_answers)
    # print(f"smart CTT mean = {mean_smart}")
    # print(f"smart CTT std = {std_smart}")
    
    # ctt_tag, ctt_t_stat, ctt_p_value = perform_t_test(
    #     dumb_answers, smart_answers, label="CTT"
    # )
    
    # # IRT via HMC
    # print("\nIRT via HMC")
    # z_easy = jnp.array(z_easy)
    # dumb_answers = jnp.array(dumb_answers)
    # theta_dumb_mean, theta_dumb_std, theta_dumb_samples = fit_theta_mcmc(
    #     z_easy, dumb_answers
    # )
    # print(f"dumb IRT mean = {theta_dumb_mean}")
    # print(f"dumb IRT std = {theta_dumb_std}")
    
    # z_hard = jnp.array(z_hard)
    # smart_answers = jnp.array(smart_answers)
    # theta_smart_mean, theta_smart_std, theta_smart_samples = fit_theta_mcmc(
    #     z_hard, smart_answers
    # )
    # print(f"smart IRT mean = {theta_smart_mean}")
    # print(f"smart IRT std = {theta_smart_std}")
    
    # irt_tag, irt_t_stat, irt_p_value = perform_t_test(
    #     theta_dumb_samples, theta_smart_samples, label="IRT"
    # )
    
    # output_df = pd.DataFrame({
    #     'ctt_tag': [ctt_tag],
    #     'ctt_t_stat': [ctt_t_stat],
    #     'ctt_p_value': [ctt_p_value],
    #     'irt_tag': [irt_tag],
    #     'irt_t_stat': [irt_t_stat],
    #     'irt_p_value': [irt_p_value]
    # })
    # output_df.to_csv(output_path, index=False)
    
    # plot_nonid_test(
    #     theta_dumb_samples,
    #     theta_smart_samples,
    #     z_easy,
    #     z_hard,
    #     plot_path
    # )
    
if __name__ == "__main__":
    # wandb.init(project="nonid_test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    # set_seed(42)
    # plot_dir = f'../plot/nonid_test'
    # output_dir = f'../data/nonid_test/{args.dataset}'
    # os.makedirs(plot_dir, exist_ok=True)
    # os.makedirs(output_dir, exist_ok=True)
    
    main(
        # theta_path=f'../data/nonamor_calibration/{args.dataset}/nonamor_theta.csv', 
        y_path=f'../data/pre_calibration/{args.dataset}/matrix.csv',
        # output_path=f'{output_dir}/nonid_test.csv',
        # plot_path=f'{plot_dir}/posttheta_zdistr_{args.dataset}.png',
    )
    