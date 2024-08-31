import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils import item_response_fn_3PL, set_seed

def model(question_num, testtaker_num, response_matrix):
    z1_hat = numpyro.sample("z1_hat", dist.Normal(0.0, 1.0).expand((question_num,)))
    z2_hat = numpyro.sample("z2_hat", dist.Normal(0.0, 1.0).expand((question_num,)))
    z3_hat = numpyro.sample("z3_hat", dist.Normal(0.0, 1.0).expand((question_num,)))
    
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0).expand((testtaker_num,)))
    
    z1_hat_expanded = jnp.expand_dims(z1_hat, 0)  # Shape: (1, question_num)
    z2_hat_expanded = jnp.expand_dims(z2_hat, 0)  # Shape: (1, question_num)
    z3_hat_expanded = jnp.expand_dims(z3_hat, 0)  # Shape: (1, question_num)
    theta_hat_expanded = jnp.expand_dims(theta_hat, 1)  # Shape: (testtaker_num, 1)
    prob_matrix = item_response_fn_3PL(
        z1_hat_expanded,
        z2_hat_expanded,
        z3_hat_expanded,
        theta_hat_expanded,
        datatype="jnp"
    )
    
    numpyro.sample("obs", dist.Bernoulli(prob_matrix), obs=response_matrix)

def irt_mcmc(question_num, testtaker_num, response_matrix, num_samples=9, num_warmup=10):
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
    z1_samples = mcmc.get_samples()["z1_hat"]
    z2_samples = mcmc.get_samples()["z2_hat"]
    z3_samples = mcmc.get_samples()["z3_hat"]

    return theta_samples, z1_samples, z2_samples, z3_samples
    
if __name__ == "__main__":
    set_seed(10)
    y_df = pd.read_csv('/Users/tyhhh/Desktop/certified-eval/data/synthetic/response_matrix/synthetic_matrix_3PL.csv', index_col=0)
    response_matrix = y_df.values
    testtaker_num, question_num = response_matrix.shape

    theta_samples, z1_samples, z2_samples, z3_samples = \
        irt_mcmc(question_num, testtaker_num, response_matrix)

    theta_samples = np.array(theta_samples) # (num_samples, testtaker_num)
    z1_samples = np.array(z1_samples) # (num_samples, question_num)
    z2_samples = np.array(z2_samples)
    z3_samples = np.array(z3_samples)

    true_theta = pd.read_csv('../data/synthetic/response_matrix/true_theta.csv')
    true_Z = pd.read_csv('../data/synthetic/response_matrix/true_Z_3PL.csv')
    true_theta = true_theta.iloc[:, 0].to_numpy()
    true_z1 = true_Z.iloc[:, 0].to_numpy()
    true_z2 = true_Z.iloc[:, 1].to_numpy()
    true_z3 = true_Z.iloc[:, 2].to_numpy()
    
    assert true_theta.shape == theta_samples[0].shape
    assert true_z1.shape == z1_samples[0].shape == z2_samples[0].shape == z3_samples[0].shape

    
    
    # Goodness of Fit
    theta = torch.tensor(true_theta, dtype=torch.float16)
    num_hmc_samples = theta_samples.shape[0]
    
    bins = np.linspace(-3, 3, 7)
    print(bins)
    # [-3. -2. -1.  0.  1.  2.  3.]
    # TODO: <-3 or >3 as new bins for real data

    diff_list = []
    for i in tqdm(range(question_num)):
        single_z1_samples = torch.tensor(z1_samples[:, i], dtype=torch.float16)
        single_z2_samples = torch.tensor(z2_samples[:, i], dtype=torch.float16)
        single_z3_samples = torch.tensor(z3_samples[:, i], dtype=torch.float16)

        y_col = y_df.iloc[:, i].values

        for j in range(len(bins) - 1):
            bin_mask = (theta >= bins[j]) & (theta < bins[j + 1])
            if bin_mask.sum() > 0: # bin not empty
                y_empirical = y_col[bin_mask].mean()

                theta_mid = (bins[j] + bins[j + 1]) / 2
                theta_mid_tensor = torch.tensor([theta_mid], dtype=torch.float16)
                y_theoretical_list = [item_response_fn_3PL(
                    single_z1_samples[k],
                    single_z2_samples[k],
                    single_z3_samples[k],
                    theta_mid_tensor
                ).item() for k in range(num_hmc_samples)]
                y_theoretical = sum(y_theoretical_list) / len(y_theoretical_list)
                diff = abs(y_empirical - y_theoretical)
                diff_list.append(diff)

    diff_array = np.array(diff_list)
    mean_diff = diff_array.mean()
    std_diff = diff_array.std()

    print(f'Mean of differences: {mean_diff}')
    print(f'Standard deviation of differences: {std_diff}')

    plt.figure(figsize=(10, 6))
    plt.hist(diff_list, bins=40, density=True, alpha=0.7, color='blue')
    plt.xlabel('Difference')
    plt.ylabel('Density')
    plt.title('Histogram of Differences (Empirical vs Theoretical)')
    plt.grid(True)
    plt.savefig(f'../plot/synthetic/MCMC_3pl.png')