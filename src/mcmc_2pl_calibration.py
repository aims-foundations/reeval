import argparse
import os
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
import pandas as pd
import wandb
from utils import item_response_fn_2PL_jnp, set_seed, goodness_of_fit_2PL

def model(question_num, testtaker_num, response_matrix):
    z2_hat = numpyro.sample("z2_hat", dist.LogNormal(0.0, 1.0).expand((question_num,)))
    z3_hat = numpyro.sample("z3_hat", dist.Normal(0.0, 1.0).expand((question_num,)))
    
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0).expand((testtaker_num,)))
    
    z2_hat_expanded = jnp.expand_dims(z2_hat, 0)  # Shape: (1, question_num)
    z3_hat_expanded = jnp.expand_dims(z3_hat, 0)  # Shape: (1, question_num)
    theta_hat_expanded = jnp.expand_dims(theta_hat, 1)  # Shape: (testtaker_num, 1)
    prob_matrix = item_response_fn_2PL_jnp(
        z2_hat_expanded,
        z3_hat_expanded,
        theta_hat_expanded,
    )
    
    numpyro.sample("obs", dist.Bernoulli(prob_matrix), obs=response_matrix)

def irt_mcmc(question_num, testtaker_num, response_matrix, num_samples=2000, num_warmup=1000):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        question_num=question_num,
        testtaker_num=testtaker_num,
        response_matrix=response_matrix,
    )
    mcmc.print_summary()
    
    theta_samples = mcmc.get_samples()["theta_hat"]
    z2_samples = mcmc.get_samples()["z2_hat"]
    z3_samples = mcmc.get_samples()["z3_hat"]

    return theta_samples, z2_samples, z3_samples
    
if __name__ == "__main__":
    wandb.init(project="mcmc_2pl_calibration")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(42)
    y_df = pd.read_csv(f'../data/pre_calibration/{args.dataset}/matrix.csv', index_col=0)
    
    response_matrix = y_df.values
    testtaker_num, question_num = response_matrix.shape

    output_dir = f'../data/mcmc_2pl_calibration/{args.dataset}'
    plot_dir = f'../plot/mcmc_2pl_calibration'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    theta_path = f'{output_dir}/theta.csv'
    z2_path = f'{output_dir}/z2.csv'
    z3_path = f'{output_dir}/z3.csv'
    
    theta_samples_path = f'{output_dir}/theta_samples.npy'
    z2_samples_path = f'{output_dir}/z2_samples.npy'
    z3_samples_path = f'{output_dir}/z3_samples.npy'
    
    if os.path.exists(theta_samples_path) and os.path.exists(z2_samples_path) and os.path.exists(z3_samples_path):
        print("Loading existing samples..")
        theta_samples = np.load(theta_samples_path)
        z2_samples = np.load(z2_samples_path)
        z3_samples = np.load(z3_samples_path)
    else:
        print("No existing file, Running MCMC..")
        theta_samples, z2_samples, z3_samples = irt_mcmc(
            question_num, testtaker_num, response_matrix
            )
        theta_samples = np.array(theta_samples) # (num_samples, testtaker_num)
        z2_samples = np.array(z2_samples) # (num_samples, question_num)
        z3_samples = np.array(z3_samples)

        np.save(theta_samples_path, theta_samples)
        np.save(z2_samples_path, z2_samples)
        np.save(z3_samples_path, z3_samples)
        
        theta_df = pd.DataFrame({'theta': theta_samples.mean(axis=0)})
        z2_df = pd.DataFrame({'z2': z2_samples.mean(axis=0)})
        z3_df = pd.DataFrame({'z3': z3_samples.mean(axis=0)})
        
        theta_df.to_csv(theta_path, index=False)
        z2_df.to_csv(z2_path, index=False)
        z3_df.to_csv(z3_path, index=False)