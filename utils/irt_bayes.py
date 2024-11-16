def mcmc(
    self,
    response_matrix,
    max_epoch: int = 10000,
):
    num_warmup = int(max_epoch * 0.2)
    num_samples = max_epoch - num_warmup

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
    # mcmc.print_summary()

    theta_samples = mcmc.get_samples()["theta_hat"]
    z3_samples = mcmc.get_samples()["z3_hat"]
    return theta_samples, z3_samples


def model(question_num, testtaker_num, response_matrix):
    z3_hat = numpyro.sample("z3_hat", dist.Normal(0.0, 1.0).expand((question_num,)))
    theta_hat = numpyro.sample(
        "theta_hat", dist.Normal(0.0, 1.0).expand((testtaker_num,))
    )

    z3_hat_expanded = jnp.expand_dims(z3_hat, 0)  # Shape: (1, question_num)
    theta_hat_expanded = jnp.expand_dims(theta_hat, 1)  # Shape: (testtaker_num, 1)
    prob_matrix = item_response_fn_1PL_jnp(
        z3_hat_expanded,
        theta_hat_expanded,
    )
    mask = response_matrix != -1
    numpyro.sample("obs", dist.Bernoulli(prob_matrix[mask]), obs=response_matrix[mask])
    # numpyro.sample("obs", dist.Bernoulli(prob_matrix), obs=response_matrix)
