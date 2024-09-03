import torch
import random
from testtaker import SimulatedTestTaker
from fit_theta import fit_theta_mcmc
import jax.numpy as jnp
from utils import item_response_fn_1PL, clear_caches, set_seed, save_state, load_state
import numpy as np
from argparse import ArgumentParser
import os

# export JAX_PLATFORM_NAME=cpu
# nohup python CAT.py > /scratch/yuhengtu/workspace/certified-eval/src/CAT.log 2>&1 &
  
def CAT_owen(z3, unasked_question_list, theta_mean):
    z3_unasked = z3[unasked_question_list]
    z3_unasked = jnp.array(z3_unasked)
    diff = jnp.abs(z3_unasked - theta_mean)
    return unasked_question_list[jnp.argmin(diff)]

def CAT_fisher(z3, unasked_question_list, theta_mean):
    fisher_info_list = []
    for unasked_question_index in unasked_question_list:
        theta = torch.tensor(theta_mean.item(), requires_grad=True)
        z_single = z3[unasked_question_index].clone().detach()    
        prob = item_response_fn_1PL(z_single, theta)
        hessian = prob * (1 - prob)
        fisher_info_list.append(hessian)
    index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
    return unasked_question_list[index_with_max_fisher_info]

def CAT_efisher(z3, unasked_question_list, theta_samples):
    theta_samples_tensor = torch.tensor(np.array(theta_samples))
    fisher_info_list = []
    for unasked_question_index in unasked_question_list:
        z_single = z3[unasked_question_index].clone().detach()
        z_single_expanded = z_single.expand(theta_samples_tensor.shape[0])
        prob = item_response_fn_1PL(z_single_expanded, theta_samples_tensor)
        hessian = prob * (1 - prob)
        fisher_info_list.append(hessian.mean())
    index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
    return unasked_question_list[index_with_max_fisher_info]
    
def main(question_num, new_testtaker, strategy, subset_question_num, warmup, state_dir):
    print(f'strategy: {strategy}')
    print(f'question_num: {question_num}')
    print(f'subset_question_num: {subset_question_num}')
    true_theta = new_testtaker.get_ability()
    
    state_path = os.path.join(state_dir, f"{strategy}_{question_num}_{true_theta}.pt")
    state = load_state(state_path)
    if state:
        z3 = state['z3']
        asked_question_list = state['asked_question_list']
        unasked_question_list = state['unasked_question_list']
        asked_answer_list = state['asked_answer_list']
        theta_means = state['theta_means']
        theta_stds = state['theta_stds']
        start_epoch = state['epoch'] + 1
    
    else:
        if warmup > 0:
            print(f'doing warmup: {warmup}')
            z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
            asked_question_list = random.sample(range(question_num), warmup)
            unasked_question_list = [i for i in range(question_num) if i not in asked_question_list]
            asked_answer_list = [new_testtaker.ask(z3, i).float() for i in asked_question_list]
            theta_means = []
            theta_stds = []
            start_epoch = 0
        
        else:
            print(f'not doing warmup')
            z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
            init_question_index = random.randint(0, question_num - 1)
            asked_question_list = [init_question_index]
            unasked_question_list = [i for i in range(question_num) if i != init_question_index]
            asked_answer_list = [new_testtaker.ask(z3, init_question_index).float()]
            theta_means = []
            theta_stds = []
            start_epoch = 0
            
    for epoch in range(start_epoch, subset_question_num):
        print(f'\nepoch: {epoch+1}')
        asked_question_jnp = jnp.array(asked_question_list)
        asked_answer_jnp = jnp.array(asked_answer_list)
        z3_jnp = jnp.array(z3)

        mean_theta, std_theta, theta_samples = fit_theta_mcmc(
            z3_jnp, 
            asked_question_jnp, 
            asked_answer_jnp
            )
        theta_means.append(mean_theta)
        theta_stds.append(std_theta)

        if len(unasked_question_list) == 0:
            break
        
        if strategy=="random":
            new_question_index = random.choice(unasked_question_list)
        elif strategy=="owen":
            new_question_index = CAT_owen(z3, unasked_question_list, mean_theta)
        elif strategy=="fisher":
            new_question_index = CAT_fisher(z3, unasked_question_list, mean_theta)
        elif strategy=="efisher":
            new_question_index = CAT_efisher(z3, unasked_question_list, theta_samples)
        else:
            raise ValueError("strategy not supported")
        
        new_answer = new_testtaker.ask(z3, new_question_index)
        unasked_question_list.remove(new_question_index)
        asked_question_list.append(new_question_index)
        asked_answer_list.append(new_answer.float())
        
        save_state(
            state_path, 
            z3 = z3,
            asked_question_list=asked_question_list, 
            unasked_question_list=unasked_question_list,
            asked_answer_list=asked_answer_list, 
            theta_means=theta_means, 
            theta_stds=theta_stds,
            epoch=epoch
            )
        
        clear_caches()
        
if __name__ == "__main__":
    # debug python CAT.py --subset_question_num 5
    
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--algo", type=str, default="fisher")
    parser.add_argument("--question_num", type=int, default=10000)
    parser.add_argument("--subset_question_num", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--true_theta", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    new_testtaker = SimulatedTestTaker(theta=args.true_theta, model="1PL")
    state_dir = "../data/synthetic/CAT"

    main(
        question_num=args.question_num, 
        new_testtaker=new_testtaker, 
        strategy=args.algo, 
        subset_question_num=args.subset_question_num, 
        warmup=args.warmup,
        state_dir = state_dir
        )
