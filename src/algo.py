import torch
from tqdm import tqdm
from utils import item_response_fn_1PL
import torch.optim as optim
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random



def amor_calibration(
    response_matrix, # response_matrix [69, 959]
    embedding, # embedding [959, 4096]
    lr_theta=0.01,
    W_init_std=5e-5,
    lr_W=5e-6,
    max_epoch=20000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta_hat = torch.normal(
        mean=0.0, std=1.0,
        size=(response_matrix.size(0),),
        requires_grad=True,
        device=device
    )
    W = torch.normal(
        mean=0.0, std=W_init_std, 
        size=(embedding.size(1),),
        requires_grad=True,
        device=device
    )
    
    optimizer_theta = optim.Adam([theta_hat], lr=lr_theta)
    optimizer_W = optim.Adam([W], lr=lr_W)
    
    losses = []
    pbar = tqdm(range(max_epoch))
    for _ in pbar:
        z_hat = torch.matmul(embedding, W) # z_hat [959]
        theta_hat_matrix = theta_hat.unsqueeze(1) # (n, 1)
        z_hat_matrix = z_hat.unsqueeze(0) # (1, m)
        prob_matrix = item_response_fn_1PL(z_hat_matrix, theta_hat_matrix)
        
        mask = response_matrix != -1
        masked_response_matrix = response_matrix.flatten()[mask.flatten()]
        masked_prob_matrix = prob_matrix.flatten()[mask.flatten()]
        
        berns = torch.distributions.Bernoulli(masked_prob_matrix)
        loss = -berns.log_prob(masked_response_matrix).mean()
        loss.backward()
        optimizer_theta.step()
        optimizer_W.step()
        optimizer_theta.zero_grad()
        optimizer_W.zero_grad()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': loss.item()})

    return theta_hat, z_hat, W, losses

