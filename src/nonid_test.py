import argparse
import os
import numpy as np
import pandas as pd
import torch
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
import wandb
from nonamor_calibration import nonamor_calibration
from utils import (
    set_seed,
    perform_t_test,
    bootstrap_mean_std, 
    item_response_fn_1PL_jnp,
    plot_nonid_test
)

def inverse_item_response_fn_1PL(y,theta):
    y = torch.tensor(y, dtype=torch.float32)
    return - theta - torch.log((1 - y) / y)

def sample_subsets(
    z: torch.Tensor, 
    dumb_theta: torch.Tensor, 
    smart_theta: torch.Tensor, 
    subset_size: int, 
    y_mean: float=0.7
):
    z_sorted, original_indices = torch.sort(z)
    std_all = z_sorted.std().item()

    mean_easy = inverse_item_response_fn_1PL(y_mean, dumb_theta).item()
    mean_hard = inverse_item_response_fn_1PL(y_mean, smart_theta).item()

    easy_probs = torch.exp(-0.5 * ((z_sorted - mean_easy) / (std_all / 2)) ** 2)
    easy_probs /= easy_probs.sum()
    hard_probs = torch.exp(-0.5 * ((z_sorted - mean_hard) / (std_all / 2)) ** 2)
    hard_probs /= hard_probs.sum()

    easy_indices_sorted = torch.multinomial(easy_probs, subset_size, replacement=False)
    hard_indices_sorted = torch.multinomial(hard_probs, subset_size, replacement=False)

    easy_indices, hard_indices = original_indices[easy_indices_sorted], original_indices[hard_indices_sorted]
    z_easy, z_hard = z[easy_indices], z[hard_indices]
    return z_easy, z_hard, easy_indices, hard_indices

def model(z_asked, answers):
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0)) # prior
    probs = item_response_fn_1PL_jnp(z_asked, theta_hat)
    numpyro.sample("obs", dist.Bernoulli(probs), obs=answers)
    
def fit_theta_mcmc(z_asked, answers, num_samples=9000, num_warmup=1000):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        z_asked=z_asked,
        answers=answers,
    )
    mcmc.print_summary()
    
    theta_samples = mcmc.get_samples()["theta_hat"]
    theta_mean, theta_std = jnp.mean(theta_samples), jnp.std(theta_samples)
    return theta_mean, theta_std, theta_samples

def main(
    theta_path, 
    y_path,
    plot_path,
    output_path
):
    theta = pd.read_csv(theta_path)['theta'].values
    y = pd.read_csv(y_path, index_col=0).values
    subset_size = min(y.shape[1] // 10, 5000)
    print(f"subset size = {subset_size}")
    
    i = np.abs(theta - 0.5).argmin()
    j = np.abs(theta - 1).argmin()
    print(f'dumb theta = {theta[i]}')
    print(f'smart theta = {theta[j]}')

    y_new = np.delete(y, [i, j], axis=0)
    _, z_new = nonamor_calibration(torch.tensor(y_new, dtype=torch.float32))
    z_easy, z_hard, easy_indices, hard_indices = sample_subsets(
        z_new.detach().cpu(),
        torch.tensor(theta[i], dtype=torch.float32), 
        torch.tensor(theta[j], dtype=torch.float32), 
        subset_size
    )
    
    dumb_answers = y[i][easy_indices]
    smart_answers = y[j][hard_indices]
    
    # CTT
    print("CTT")
    mean_dumb, std_dumb = bootstrap_mean_std(dumb_answers)
    print(f"dumb CTT mean = {mean_dumb}")
    print(f"dumb CTT std= {std_dumb}")
    
    mean_smart, std_smart = bootstrap_mean_std(smart_answers)
    print(f"smart CTT mean = {mean_smart}")
    print(f"smart CTT std = {std_smart}")
    
    ctt_tag, ctt_t_stat, ctt_p_value = perform_t_test(dumb_answers, smart_answers, label="CTT")
    
    # IRT via HMC
    print("\nIRT via HMC")
    z_easy = jnp.array(z_easy)
    dumb_answers = jnp.array(dumb_answers)
    theta_dumb_mean, theta_dumb_std, theta_dumb_samples = fit_theta_mcmc(z_easy, dumb_answers)
    print(f"dumb IRT mean = {theta_dumb_mean}")
    print(f"dumb IRT std = {theta_dumb_std}")
    
    z_hard = jnp.array(z_hard)
    smart_answers = jnp.array(smart_answers)
    theta_smart_mean, theta_smart_std, theta_smart_samples = fit_theta_mcmc(z_hard, smart_answers)
    print(f"smart IRT mean = {theta_smart_mean}")
    print(f"smart IRT std = {theta_smart_std}")
    
    irt_tag, irt_t_stat, irt_p_value = perform_t_test(theta_dumb_samples, theta_smart_samples, label="IRT")
    
    output_df = pd.DataFrame({
        'ctt_tag': [ctt_tag],
        'ctt_t_stat': [ctt_t_stat],
        'ctt_p_value': [ctt_p_value],
        'irt_tag': [irt_tag],
        'irt_t_stat': [irt_t_stat],
        'irt_p_value': [irt_p_value]
    })
    output_df.to_csv(output_path, index=False)
    
    plot_nonid_test(
        theta_dumb_samples,
        theta_smart_samples,
        z_easy,
        z_hard,
        plot_path
    )
    
if __name__ == "__main__":
    wandb.init(project="nonid_test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    plot_dir = f'../plot/nonid_test'
    output_dir = f'../data/nonid_test/{args.dataset}'
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    main(
        theta_path=f'../data/nonamor_calibration/{args.dataset}/nonamor_theta.csv', 
        y_path=f'../data/pre_calibration/{args.dataset}/matrix.csv',
        output_path=f'{output_dir}/nonid_test.csv',
        plot_path=f'{plot_dir}/posttheta_zdistr_{args.dataset}.png',
    )