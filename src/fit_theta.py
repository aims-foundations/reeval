import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from utils import item_response_fn_1PL
from numpyro.diagnostics import hpdi
import torch
import torch.optim as optim
from utils import item_response_fn_3PL, item_response_fn_2PL, item_response_fn_1PL
from testtaker import SimulatedTestTaker
import matplotlib.pyplot as plt
import seaborn as sns

def fit_theta_mle(Z, asked_question_list, asked_answer_list, epoch=300):
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(1,), requires_grad=True)
    optimizer = optim.Adam([theta_hat], lr=0.01)
    for _ in range(epoch):
        log_prob = 0
        for asked_question_index in asked_question_list:
            # prob = item_response_fn_3PL(*Z[asked_question_index, :], theta_hat)
            prob = item_response_fn_1PL(Z[asked_question_index], theta_hat)
            bernoulli = torch.distributions.Bernoulli(prob)
            log_prob = log_prob + bernoulli.log_prob(asked_answer_list[asked_question_index].float())
        
        loss = -log_prob/len(asked_question_list)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(theta_hat)
        
    return theta_hat

def model(Z, asked_question_list, asked_answer_list):
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0)) # prior
    Z_asked = Z[asked_question_list]
    probs = item_response_fn_1PL(Z_asked, theta_hat, datatype="jnp")
    numpyro.sample("obs", dist.Bernoulli(probs), obs=asked_answer_list)

# def model(asked_question_list, asked_answer_list):
#     Z_asked = numpyro.sample("Z", dist.Normal(0.0, 1.0, (asked_question_list.size(),)))
#     theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0)) # prior
#     Z_asked = Z[asked_question_list]
#     probs = item_response_fn_1PL(Z_asked, theta_hat, datatype="jnp")
#     numpyro.sample("obs", dist.Bernoulli(probs), obs=asked_answer_list)

def fit_theta_mcmc(Z, asked_question_list, asked_answer_list, num_samples=9000, num_warmup=1000):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        Z=Z,
        asked_question_list=asked_question_list,
        asked_answer_list=asked_answer_list,
    )
    mcmc.print_summary()
    
    theta_samples = mcmc.get_samples()["theta_hat"]
    mean_theta = jnp.mean(theta_samples)
    std_theta = jnp.std(theta_samples)

    return mean_theta, std_theta, theta_samples

def fit_theta_mcmc_key(key, Z, asked_question_list, asked_answer_list, num_samples=9000, num_warmup=1000):
    rng_key = random.PRNGKey(key)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        Z=Z,
        asked_question_list=asked_question_list,
        asked_answer_list=asked_answer_list,
    )
    mcmc.print_summary()
    
    theta_samples = mcmc.get_samples()["theta_hat"]
    mean_theta = jnp.mean(theta_samples)
    std_theta = jnp.std(theta_samples)
    # hpdi_theta = hpdi(theta_samples, 0.9)

    return mean_theta, std_theta, theta_samples

def plot_trace_and_density(theta_samples_list):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    for i, theta_samples in enumerate(theta_samples_list):
        plt.plot(theta_samples, label=f'Run {i+1}', alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('θ')
    plt.title('Trace Plot of θ')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i, theta_samples in enumerate(theta_samples_list):
        sns.kdeplot(theta_samples, bw_adjust=0.5, label=f'Run {i+1}', alpha=0.3)
    plt.xlabel('θ')
    plt.title('Posterior Density of θ')
    plt.legend()

    plt.tight_layout()
    plt.savefig("../plot/synthetic/mcmc_diagnostic_2.png")
    
if __name__ == "__main__":
    torch.manual_seed(10)
    question_num = 1000

    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    new_testtaker = SimulatedTestTaker(model="1PL")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")

    asked_question_list = list(range(question_num))
    asked_answer_list = []
    for i in range(question_num):
        asked_answer_list.append(new_testtaker.ask(z3, i))
    
    # MLE
    # theta_hat = fit_theta_mle(z3, asked_question_list, asked_answer_list, epoch=300)
    # print(f"mle theta: {theta_hat}")

    # MCMC
    asked_question_list = jnp.array(asked_question_list)
    asked_answer_list = jnp.array(asked_answer_list)
    z3 = jnp.array(z3)

    mean_theta, std_theta, theta_samples = fit_theta_mcmc(z3, asked_question_list, asked_answer_list)

    print(theta_samples)
    
    print(f"mcmc theta mean: {mean_theta}")
    print(f"mcmc theta std: {std_theta}")
    
    theta_samples_list = [theta_samples[:1000], theta_samples[:1001]]
    plot_trace_and_density(theta_samples_list)
