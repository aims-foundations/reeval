import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from utils import item_response_fn_1PL
import torch
from utils import set_seed, item_response_fn_1PL
from synthetic_testtaker import SimulatedTestTaker

def model(question_num, testtaker_num, response_matrix):
    Z_hat = numpyro.sample("Z", dist.Normal(0.0, 1.0, (question_num,)))
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0), (testtaker_num,))
    
    Z_hat_expanded = jnp.expand_dims(Z_hat, 0)  # Shape: (1, question_num)
    theta_hat_expanded = jnp.expand_dims(theta_hat, 1)  # Shape: (testtaker_num, 1)
    prob_matrix = item_response_fn_1PL(Z_hat_expanded, theta_hat_expanded, datatype="jnp")
    
    numpyro.sample("obs", dist.Bernoulli(prob_matrix), obs=response_matrix)

def fit_theta_mcmc(question_num, testtaker_num, response_matrix, num_samples=9000, num_warmup=1000):
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
    mean_theta = jnp.mean(theta_samples)
    std_theta = jnp.std(theta_samples)

    Z_samples = mcmc.get_samples()["Z"]
    mean_Z = jnp.mean(Z_samples)
    std_Z = jnp.std(Z_samples)
    
    return mean_theta, std_theta, theta_samples, mean_Z, std_Z, Z_samples
    
if __name__ == "__main__":
    set_seed(10)
    response_matrix = 

    mean_theta, std_theta, theta_samples, mean_Z, std_Z, Z_samples = \
        fit_theta_mcmc(z3, asked_question_list, asked_answer_list)

    print(f"mcmc theta mean: {mean_theta}")
    print(f"mcmc theta std: {std_theta}")
    