import torch
from tqdm import tqdm
from torch.optim import LBFGS
from torch.distributions import Bernoulli
def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
    return parameters

# should return probs
def rasch(data_with0, train_idtor, B=50_000, device="cuda:0"):
    optimized_zs = []
    n_test_takers, n_items = data_with0.shape[0], data_with0.shape[1]
    data_with0 = data_with0.to(device)
    train_idtor = train_idtor.to(device)
    thetas_nuisance = torch.randn(150, n_test_takers, device=device)
    for i in tqdm(range(0, n_items, B)):
        data_batch = data_with0[:, i:i+B]
        train_idtor_batch = train_idtor[:, i:i+B]
        current_B = data_batch.shape[1]
        z_i = torch.randn(current_B, requires_grad=True, device=device)
        optim_z_i = LBFGS([z_i], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_z_i():
            optim_z_i.zero_grad()
            probs = torch.sigmoid(thetas_nuisance[:, :, None] + z_i[None, None, :])
            loss = -(Bernoulli(probs=probs).log_prob(data_batch)*train_idtor_batch).mean()
            loss.backward()
            return loss
        z_i_optimized = trainer([z_i], optim_z_i, closure_z_i)[0].detach()
        optimized_zs.append(z_i_optimized)
    zs = torch.cat(optimized_zs)

    # fit theta
    thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
    optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_theta():
        optim_theta.zero_grad()
        probs = torch.sigmoid(thetas[:, None] + zs[None, :])
        loss = -(Bernoulli(probs=probs).log_prob(data_with0)*train_idtor).mean()
        loss.backward()
        return loss
    thetas = trainer([thetas], optim_theta, closure_theta)[0]

    # calculate metrics
    probs = torch.sigmoid(thetas[:, None] + zs[None, :])
    
    return probs, thetas, zs
