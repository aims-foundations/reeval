import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
import pandas as pd
from utils import item_response_fn_1PL, compute_mse
from utils import set_seed, item_response_fn_1PL

def model(question_num, testtaker_num, response_matrix):
    Z_hat = numpyro.sample("Z", dist.Normal(0.0, 1.0).expand((question_num,)))
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0).expand((testtaker_num,)))
    
    Z_hat_expanded = jnp.expand_dims(Z_hat, 0)  # Shape: (1, question_num)
    theta_hat_expanded = jnp.expand_dims(theta_hat, 1)  # Shape: (testtaker_num, 1)
    prob_matrix = item_response_fn_1PL(Z_hat_expanded, theta_hat_expanded, datatype="jnp")
    
    numpyro.sample("obs", dist.Bernoulli(prob_matrix), obs=response_matrix)

def irt_mcmc(question_num, testtaker_num, response_matrix, num_samples=9000, num_warmup=1000):
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
    Z_samples = mcmc.get_samples()["Z"]
    
    return theta_samples, Z_samples
    
if __name__ == "__main__":
    set_seed(10)
    response_matrix = pd.read_csv('../data/synthetic/response_matrix/synthetic_matrix.csv', index_col=0).values
    testtaker_num, question_num = response_matrix.shape

    theta_samples, Z_samples = irt_mcmc(question_num, testtaker_num, response_matrix)
    
    theta_mean = jnp.mean(theta_samples, axis=0)
    Z_mean = jnp.mean(Z_samples, axis=0)

    theta_mean = np.array(theta_mean)
    Z_mean = np.array(Z_mean)

    true_theta = pd.read_csv('/Users/tyhhh/Desktop/certified-eval/data/synthetic/response_matrix/true_theta.csv')
    true_Z = pd.read_csv('/Users/tyhhh/Desktop/certified-eval/data/synthetic/response_matrix/true_Z.csv')
    true_theta = true_theta.iloc[:, 0].to_numpy()
    true_Z = true_Z.iloc[:, 0].to_numpy()
    
    assert true_theta.shape == theta_mean.shape
    assert true_Z.shape == Z_mean.shape
    
    theta_mse = compute_mse(theta_mean, true_theta)
    Z_mse = compute_mse(Z_mean, true_Z)

    print(f"Theta MSE: {theta_mse}")
    print(f"Z MSE: {Z_mse}")
